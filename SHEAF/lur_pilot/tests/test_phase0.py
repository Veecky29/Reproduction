"""No-training checks for the Phase 0 driver.

These tests do not import torch and do not call the official trainers.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import phase0


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _value_header(n_values: int, extra: list[str] | None = None) -> list[str]:
    """date + (n_values - 1) plain columns + OT, then optional extras inserted before OT."""
    extras = extra or []
    plain = n_values - 1 - len(extras)
    if plain < 0:
        raise ValueError("n_values too small for the extra columns")
    return ["date", *[f"c{i}" for i in range(plain)], *extras, "OT"]


class RainColumns(unittest.TestCase):
    def test_names(self) -> None:
        self.assertTrue(phase0.is_rain_column("rain (mm)"))
        self.assertTrue(phase0.is_rain_column("raining (s)"))
        self.assertTrue(phase0.is_rain_column("rain"))
        self.assertTrue(phase0.is_rain_column("Raining"))
        self.assertTrue(phase0.is_rain_column("  RAIN (MM) "))
        self.assertFalse(phase0.is_rain_column("T (degC)"))
        self.assertFalse(phase0.is_rain_column("rainfall"))
        self.assertFalse(phase0.is_rain_column("max. wv (m/s)"))
        self.assertFalse(phase0.is_rain_column("OT"))


class Selection(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = phase0.load_config(phase0.ROOT / "config.yaml")
        self.parser = phase0.build_parser()

    def test_default_excludes_ett(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(
            phase0.selected_datasets(self.cfg, args),
            ["electricity", "traffic", "weather"],
        )
        self.assertFalse(self.cfg["include_ett"])
        self.assertEqual(self.cfg["horizon"], 96)

    def test_include_ett_flag(self) -> None:
        args = self.parser.parse_args(["--include-ett"])
        names = phase0.selected_datasets(self.cfg, args)
        self.assertEqual(names[:3], ["electricity", "traffic", "weather"])
        self.assertEqual(names[3:], list(phase0.ETT_DATASETS))

    def test_explicit_subset(self) -> None:
        args = self.parser.parse_args(["--dataset", "weather"])
        self.assertEqual(phase0.selected_datasets(self.cfg, args), ["weather"])


class Prepare(unittest.TestCase):
    def test_missing_traffic_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(root / "electricity.csv", ["date", "OT"], [["2000-01-01 00:00:00", "1"]])
            with self.assertRaises(SystemExit) as ctx:
                phase0.resolve_csv(root, "traffic")
        message = str(ctx.exception)
        self.assertIn("does not skip", message)
        self.assertIn("traffic.csv", message)
        self.assertIn("Traffic.csv", message)

    def test_weather_drops_rain_without_rewriting_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "src"
            dest = Path(tmp) / "prepared"
            header = _value_header(21, ["rain (mm)", "raining (s)"])
            self.assertEqual(len(header) - 1, 21)
            row = ["2010-01-01 00:10:00", *["0.10"] * (len(header) - 4), "1.50", "30", "9.25"]
            self.assertEqual(len(row), len(header))
            _write_csv(root / "weather" / "weather.csv", header, [row])
            source = phase0.resolve_csv(root, "weather")
            record = phase0.prepare_one(
                "weather", source, dest, target="OT", drop_rain=True, allow_channel_mismatch=False
            )
            self.assertEqual(record["columns_dropped"], ["rain (mm)", "raining (s)"])
            self.assertEqual(record["enc_in"], 19)
            self.assertFalse(record["drop_is_method"])
            prepared = phase0.read_header(dest / "weather.csv")
            self.assertNotIn("rain (mm)", prepared)
            self.assertNotIn("raining (s)", prepared)
            with (dest / "weather.csv").open(newline="") as handle:
                body = list(csv.reader(handle))[1]
            self.assertIn("0.10", body)
            self.assertIn("9.25", body)
            self.assertNotIn("1.50", body)
            self.assertNotIn("30", body)

    def test_unrecognized_21_column_weather_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = _value_header(21)
            _write_csv(root / "weather.csv", header, [["2010-01-01 00:10:00", *["1"] * 21]])
            with self.assertRaises(SystemExit) as ctx:
                phase0.prepare_one(
                    "weather",
                    root / "weather.csv",
                    root / "out",
                    target="OT",
                    drop_rain=True,
                    allow_channel_mismatch=False,
                )
        self.assertIn("rain", str(ctx.exception))


class Settings(unittest.TestCase):
    def _spec(self, dataset: str, model: str, enc_in: int) -> phase0.RunSpec:
        manifest = {
            "horizon": 96,
            "datasets": {
                dataset: {
                    "enc_in": enc_in,
                    "prepared": f"/data/{phase0.DATA_FILE[dataset]}",
                    "data_path": phase0.DATA_FILE[dataset],
                    "data_key": phase0.DATA_KEY[dataset],
                    "target": "OT",
                    "freq": "h",
                }
            },
        }
        cfg = phase0.load_config(phase0.ROOT / "config.yaml")
        args = phase0.build_parser().parse_args([])
        return phase0.make_spec(dataset, model, manifest, cfg, args)

    def test_official_setting_strings(self) -> None:
        itransformer = self._spec("electricity", "iTransformer", 321)
        self.assertEqual(
            itransformer.setting(),
            "ECL_96_96_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_"
            "fctimeF_ebTrue_dtExp_psloss0_projection_0",
        )
        dlinear = self._spec("electricity", "DLinear", 321)
        self.assertEqual(
            dlinear.setting(),
            "Electricity_96_96_DLinear_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_"
            "fc1_ebtimeF_dtTrue_psloss0_Exp_0",
        )

    def test_argv_matches_upstream_choices(self) -> None:
        electricity = self._spec("electricity", "iTransformer", 321)
        argv = electricity.argv(is_training=1)
        self.assertIn("--use_ps_loss", argv)
        self.assertEqual(argv[argv.index("--use_ps_loss") + 1], "0")
        self.assertEqual(argv[argv.index("--e_layers") + 1], "3")
        self.assertEqual(argv[argv.index("--enc_in") + 1], "321")
        self.assertEqual(argv[argv.index("--batch_size") + 1], "16")
        self.assertEqual(argv[argv.index("--learning_rate") + 1], "0.0005")
        self.assertNotIn("--individual", argv)

        weather = self._spec("weather", "iTransformer", 19)
        wargv = weather.argv(is_training=1)
        self.assertEqual(wargv[wargv.index("--enc_in") + 1], "19")
        self.assertEqual(wargv[wargv.index("--data_path") + 1], "weather.csv")
        self.assertNotIn("--batch_size", wargv)
        self.assertEqual(wargv[wargv.index("--learning_rate") + 1], "0.0001")

        traffic = self._spec("traffic", "iTransformer", 862)
        targv = traffic.argv(is_training=1)
        self.assertEqual(targv[targv.index("--e_layers") + 1], "4")
        self.assertEqual(targv[targv.index("--enc_in") + 1], "862")
        self.assertEqual(targv[targv.index("--seq_len") + 1], "96")
        self.assertEqual(targv[targv.index("--pred_len") + 1], "96")

        dweather = self._spec("weather", "DLinear", 19)
        dargv = dweather.argv(is_training=1)
        self.assertNotIn("--e_layers", dargv)
        self.assertEqual(dargv[dargv.index("--batch_size") + 1], "16")
        self.assertEqual(dargv[dargv.index("--learning_rate") + 1], "0.0001")

    def test_horizon_flag_changes_both_lengths(self) -> None:
        manifest = {
            "horizon": 192,
            "datasets": {
                "electricity": {
                    "enc_in": 321,
                    "prepared": "/data/electricity.csv",
                    "data_path": "electricity.csv",
                    "data_key": "custom",
                    "target": "OT",
                    "freq": "h",
                }
            },
        }
        cfg = phase0.load_config(phase0.ROOT / "config.yaml")
        args = phase0.build_parser().parse_args(["--horizon", "192"])
        spec = phase0.make_spec("electricity", "DLinear", manifest, cfg, args)
        self.assertEqual(spec.seq_len, 192)
        self.assertEqual(spec.horizon, 192)
        self.assertIn("Electricity_192_192", spec.model_id)


class KillRule(unittest.TestCase):
    def _band(self, lur: float, weight: float = 0.2) -> list[dict]:
        return [{"band": "[0.000,0.050)", "lur_mean": lur, "weight_mean": weight, "n_channels": 16}]

    def test_weather_only_stops(self) -> None:
        verdict = phase0.kill_verdict(
            {
                "electricity": self._band(0.9),
                "traffic": self._band(0.8),
                "weather": self._band(0.1),
            },
            protocol_ok=True,
            xr_below=0.5,
            min_weight=0.05,
        )
        self.assertEqual(verdict["verdict"], "stop")
        self.assertIn("Weather", verdict["text"])

    def test_low_band_on_electricity_does_not_stop(self) -> None:
        verdict = phase0.kill_verdict(
            {
                "electricity": self._band(0.2),
                "traffic": self._band(0.9),
            },
            protocol_ok=True,
            xr_below=0.5,
            min_weight=0.05,
        )
        self.assertEqual(verdict["verdict"], "not_stopped")

    def test_low_energy_band_is_not_material(self) -> None:
        verdict = phase0.kill_verdict(
            {
                "electricity": self._band(0.1, weight=0.01),
                "traffic": self._band(0.9),
            },
            protocol_ok=True,
            xr_below=0.5,
            min_weight=0.05,
        )
        self.assertEqual(verdict["verdict"], "stop")

    def test_missing_traffic_is_incomplete(self) -> None:
        verdict = phase0.kill_verdict(
            {"electricity": self._band(0.9)},
            protocol_ok=True,
            xr_below=0.5,
            min_weight=0.05,
        )
        self.assertEqual(verdict["verdict"], "incomplete")

    def test_changed_protocol_is_not_a_stop(self) -> None:
        verdict = phase0.kill_verdict(
            {
                "electricity": self._band(0.9),
                "traffic": self._band(0.9),
            },
            protocol_ok=False,
            xr_below=0.5,
            min_weight=0.05,
        )
        self.assertEqual(verdict["verdict"], "not_preregistered")


class Channels(unittest.TestCase):
    def test_even_includes_ends(self) -> None:
        chosen = phase0.even_channels(321, 16)
        self.assertEqual(chosen[0], 0)
        self.assertEqual(chosen[-1], 320)
        self.assertEqual(len(chosen), 16)
        self.assertEqual(phase0.parse_channel_spec("all", 5), [0, 1, 2, 3, 4])
        self.assertEqual(phase0.parse_channel_spec("0,2", 5), [0, 2])


class Protocol(unittest.TestCase):
    def test_shipped_config_matches_preregistration(self) -> None:
        cfg = phase0.load_config(phase0.ROOT / "config.yaml")
        view = {
            "horizon": cfg["horizon"],
            "datasets": tuple(cfg["datasets"]),
            "kill_model": cfg["kill_rule"]["model"],
            "kill_datasets": tuple(cfg["kill_rule"]["datasets"]),
            "lur_bands": cfg["lur_bands"],
            "lur_max_samples": cfg["lur_max_samples"],
            "lur_channels": cfg["lur_channels"],
            "lur_include_dc": cfg["lur_include_dc"],
            "welch_win_frac": cfg["welch_win_frac"],
            "welch_overlap": cfg["welch_overlap"],
            "alpha_boundary": cfg["alpha_boundary"],
            "xr_below": cfg["kill_rule"]["xr_below"],
            "min_energy_weight": cfg["kill_rule"]["min_energy_weight"],
            "weather_drop_rain": cfg["weather_drop_rain"],
        }
        self.assertTrue(phase0.protocol_is_preregistered(view))


if __name__ == "__main__":
    unittest.main()
