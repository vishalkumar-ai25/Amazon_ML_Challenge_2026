#!/bin/bash
# Auto-switch script: monitors US completion, kills old process, launches India-fast
cd ~/Amazon_ML_Challenge_2026

US_TOTAL=664
LOG_FILE=logs/auto_switch.log

echo "[$(date)] Auto-switch monitor started. Waiting for US to complete ($US_TOTAL shards)..." >> $LOG_FILE

while true; do
    US_DONE=$(ls output/parts_v4/shards/claim_US_*.csv 2>/dev/null | wc -l)
    echo "[$(date)] US shards: $US_DONE/$US_TOTAL" >> $LOG_FILE
    
    if [ "$US_DONE" -ge "$US_TOTAL" ]; then
        echo "[$(date)] US COMPLETE! All $US_TOTAL shards done." >> $LOG_FILE
        
        # Kill the old inference_turbo process
        OLD_PID=$(ps aux | grep 'inference_turbo.py' | grep -v grep | grep -v auto_switch | awk '{print $2}' | head -1)
        if [ -n "$OLD_PID" ]; then
            echo "[$(date)] Killing old inference_turbo.py (PID $OLD_PID)" >> $LOG_FILE
            kill $OLD_PID 2>/dev/null
            sleep 5
            kill -9 $OLD_PID 2>/dev/null
            sleep 3
        fi
        
        echo "[$(date)] Launching India-fast inference..." >> $LOG_FILE
        nohup venv/bin/python3 src/inference_india_fast.py --n-workers 32 > logs/inference_india_fast.log 2>&1 &
        echo "[$(date)] India-fast launched with PID $!" >> $LOG_FILE
        exit 0
    fi
    
    sleep 30
done
