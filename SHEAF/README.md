# SHEAF 鈥?Statistical Hypothesis鈥揈dge Attribution Framework

> 2026-11-26 璧风殑宸ヤ綔鍖恒€?*缁撹鏄惁瀹氱殑**锛氱簿纭彲鍒犻櫎鐨勫姞鎬ц鍑哄眰鍦ㄨ繖鎵瑰熀鍑嗕笂
> 娌℃湁甯︽潵浠讳綍棰勬祴澧炵泭銆傝 [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md)銆?
## 涓€鍙ヨ瘽

鍐荤粨涓€涓?channel-independent 楠ㄥ共锛岃**绐楀彛缁熻閲?*閫氳繃涓€涓?*鎸夛紙鐩爣 脳 棰勬祴鍧?脳 婧愶級
鍙姞鐨勮鍑哄眰**杩涘叆棰勬祴锛屼簬鏄?绗?j 涓彉閲忕殑缁熻锛屾槸鍚︺€佸湪鍝釜棰勬祴姝ヤ笂锛屽绗?i 涓洰鏍囨湁鐢?
鍙樻垚涓€涓兘琚垹闄ゅ疄楠岃瘉浼殑闂銆?
杩欎釜鎯虫硶鏈韩鏄共鍑€鐨勩€?*瀹冨け璐ュ湪鏁版嵁涓?*锛氳兘鍚冨埌璺ㄩ€氶亾淇℃伅鐨勫湴鏂癸紙Electricity/Weather锛夛紝
鑷敱閫氶亾娣峰悎姣斿畠濂?13鈥?2%锛涗笉鑳藉悆鍒扮殑鍦版柟锛圗TT锛夛紝鎵€鏈夋柟娉曢兘鎵撳钩銆?
## 涓や釜闃舵

### 闃舵涓€锛氭鏌?B 鈥斺€?杈瑰厛楠?蟻 鍙俊鍚?
`checkB/`銆傜敤 6 涓爣鍑嗗熀鍑嗘楠?绐楀彛鍐呮粸鍚庣浉鍏?蟻"鑳藉惁棰勬祴鎴愬渚濊禆锛堟粸鍚庝簰淇℃伅锛夈€?
涓昏淇锛堣繃绋嬩腑鍙戠幇鏃х粨璁烘湁璇級锛?
- **鏃х殑 `pair_stats()` 绠楃殑涓嶆槸鐩稿叧绯绘暟**锛氬畠鎶婅法绐楀彛璺?lag 鐨勫叏灞€鏈€澶?|cc| 闄や互
  `200脳13`锛屽€煎煙 3e-4銆傛敼鎴愮湡 Pearson 鍚庯紝`蟻~MI` 浠?0.291 鍙樻垚 0.542锛圗TTh1锛夈€?- **"閫愮獥鍙?蟻 浼樹簬鍏ㄥ簭鍒?蟻" 鏄弽鐨?*锛?.77 鎵嶆槸鍏ㄥ簭鍒楋紝0.29 鏄€愮獥鍙ｃ€?- **蟻 蹇呴』鍦ㄥ幓瓒嬪娍+鍘诲鑺傜殑娈嬪樊涓婄畻**锛氬師濮嬪煙涓?ECL 鏈?15% 鐨勮竟 蟻鈮?.90锛?  鏄獥鍙ｅ唴瓒嬪娍閫犳垚鐨勪吉鐩稿叧銆俽esid 鍩熶笂 蟻~MI = 0.69鈥?.84锛屼笉鍐嶉ケ鍜屻€?
### 闃舵浜岋細涓夎噦瀵圭収 鈥斺€?鍙垹闄よ鍑哄眰鏈夌敤鍚?
`w12/`銆傚悓涓€缂栫爜鍣ㄣ€佸悓涓€鍗忚涓嬪鐓?`plain` / `concat` / `sheaf`锛? 涓暟鎹泦 脳 3 绉嶅瓙
锛圱raffic 鏈窇閫氾紝鍘熷洜瑙佽礋缁撴灉 搂2 鐨勬敞锛夈€?
缁撹锛?*`sheaf` 鈮?`plain`锛埪?%锛夛紝`concat` 鍦?Electricity 涓婃瘮瀹冧滑濂?17.7%銆?*

### 闃舵涓夛細鍙紑鍏崇殑璺ㄩ€氶亾娣峰悎锛堜篃澶辫触锛?
`MIXING_SWITCH.md`銆傛棦鐒跺叕骞冲鐓т笅娣峰悎鍙湪 Electricity 涓婂€?9.4%銆佸湪 ETT 涓婃湁瀹筹紝
灏辫瘯浜?`h'_i = h_i + 尾_i路Mix(h)_i`锛岃妯″瀷鑷繁瀛﹀紑鍏炽€?
**尾 鍦ㄤ袱涓柟鍚戦兘涓嶅姩**锛?.1 璧锋鍋滃湪 0.1锛?.0 璧锋鍋滃湪 0.97锛夛紝
涓?尾=1 涔熶笉绛変簬绋犲瘑娣峰悎锛堜綆绉╃摱棰堣ˉ涓嶅洖鏉ワ級銆傝繖鏉′篃鏀跺彛銆?
## 鐩綍

```
SHEAF/
  NEGATIVE_RESULT.md        # 璐熺粨鏋滐細璁捐銆佹暟瀛椼€佷笁涓崗璁潙
  HANDOFF.md                # 椤圭洰浜ゆ帴锛堝巻鍙插喅绛栦笌淇璁板綍锛?  README.md
  checkB/                   # 闃舵涓€锛氳竟鍏堥獙 蟻 鐨勫彲闈犳€?    checkB.py               # 涓昏剼鏈紙鏃х殑 蟻 浼拌鍣紝淇濈暀浣滃鐓э級
    checkB_detail.py
    checkB_partial.py
    checkB_deseason.py      # 鍘昏秼鍔?鍘诲鑺?+ 鍚屽煙閲嶇畻 MI
    checkB_estimator.py     # 鏂版棫浼拌鍣ㄥ鐓?+ bootstrap CI
    checkB_scope.py         # 2脳2锛氬叏灞€/閫愮獥 脳 鍘熷/娈嬪樊
    results/*.csv, *.log
  w12/                      # 闃舵浜岋細涓夎噦瀵圭収
    data.py                 # 鍒囧垎銆佹粦绐椼€乺esid 鍩?蟻 鍏堥獙
    models.py               # plain / concat / sheaf
    train.py                # 鍗曢厤缃缁?    summarize.py            # 姹囨€?    driver*.py, run_all.sh  # 闃熷垪
    results/*.jsonl
  docs/report_review.md     # 瀵瑰垵鐗?report.docx 鐨勮瘎瀹★紙鍘嗗彶瀛樻。锛?```

## 澶嶇幇

鏁版嵁涓嶈繘 git锛?36MB 鐨?Traffic 绛夛級銆傛寜 `data/SHA256SUMS` 鏍￠獙鍘熷 CSV锛屾斁鍦ㄥ悓涓€鐩綍锛?
```bash
# 闃舵涓€
cd SHEAF/checkB && python checkB_deseason.py

# 闃舵浜岋紙闇€瑕?GPU锛?cd SHEAF/w12 && python train.py --dataset Electricity --arm sheaf --horizon 96 \
    --lr 3e-4 --channels 9999 --stride 4 --batch 32
```

鏁版嵁婧愶紙涓?TSLib 瀹樻柟涓€鑷达級锛?
- ETT锛歚https://raw.githubusercontent.com/zhouhaoyi/ETDataset/.../ETT-small/ETT*.csv`
- Weather / Electricity / Traffic锛?  `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59.../<name>/<name>.csv`

鍒囧垎蹇呴』鐢?TSLib 鐨?30 澶╂湀杈圭晫锛坄8640/11520/14400`锛夛紝涓嶆槸鏃ュ巻鏈堛€?璇﹁ [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) 搂4.1銆?
## 鐩稿鏃㈡湁宸ヤ綔鐨勪綅缃?
鍚屾湡鐨?DAED銆丳atchDecomp銆乀empShap銆両B-Forecast 宸茬粡瑕嗙洊浜?棰勬祴鐨勫彲鍔犲垎瑙?涓?"蹇犲疄搴︿繚璇?锛屾湰宸ヤ綔涓庡畠浠殑宸埆鍙湪鍒嗚В缁村害锛堟簮 脳 鐩爣 脳 棰勬祴鍧楋級鍜屾棤 softmax 绔炰簤銆?鍦ㄧ簿搴︿笂鏈伐浣滄病鏈変紭鍔匡紝鍥犳杩欐潯绾夸笉鏋勬垚鍙彂琛ㄧ殑澧為噺銆?
