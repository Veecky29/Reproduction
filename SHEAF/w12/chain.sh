#!/usr/bin/env bash
# wait for driver2 to finish, then run the lr search on plain
while pgrep -f driver2.py > /dev/null; do sleep 60; done
cd /root/autodl-tmp/sheaf_w12
/root/miniconda3/bin/python driver3.py >> logs/driver3.log 2>&1
# then append traffic to the main matrix
cd /root/autodl-tmp/sheaf_w12
/root/miniconda3/bin/python driver2.py >> logs/driver2b.log 2>&1
echo CHAINDONE >> logs/chain.log
