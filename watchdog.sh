#!/bin/bash
# Watchdog for FSD evaluation pipeline - monitors and restarts if crashed

SCRIPT_DIR=/data/awais/projects/FSD
PID_FILE=$SCRIPT_DIR/logs/pid.txt
LOG_FILE=$SCRIPT_DIR/logs/watchdog.log
PYTHON=/data/awais/anaconda/envs/fsd/bin/python
CUDNN_LIB=/data/awais/anaconda/envs/fsd/lib/python3.12/site-packages/nvidia/cudnn/lib

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

start_pipeline() {
    cd $SCRIPT_DIR
    export CUDA_VISIBLE_DEVICES=5,6,7
    export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
    nohup $PYTHON evaluate_datasets.py >> $SCRIPT_DIR/logs/main_run.log 2>&1 &
    PID=$!
    echo $PID > $PID_FILE
    log "Started pipeline with PID $PID"
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if kill -0 $PID 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

check_progress() {
    # Check if pipeline is making progress by monitoring CSV growth
    AIGI_COUNT=$(wc -l < $SCRIPT_DIR/Results/AIGI_TEST/predictions.csv 2>/dev/null || echo 0)
    EVAL_COUNT=$(wc -l < $SCRIPT_DIR/Results/image_eval24/predictions.csv 2>/dev/null || echo 0)
    REWIND_COUNT=$(wc -l < $SCRIPT_DIR/Results/ReWIND/predictions.csv 2>/dev/null || echo 0)

    # Check if final report exists (all done)
    if [ -f "$SCRIPT_DIR/Results/final_report.txt" ]; then
        log "Final report found - pipeline completed!"
        return 2
    fi

    echo "$AIGI_COUNT $EVAL_COUNT $REWIND_COUNT"
    return 0
}

log "Watchdog started"

LAST_COUNT=0
STALE_COUNT=0

while true; do
    if is_running; then
        # Check progress every 5 minutes
        COUNTS=$(check_progress)
        STATUS=$?

        if [ $STATUS -eq 2 ]; then
            log "Pipeline completed successfully!"
            break
        fi

        TOTAL_COUNT=$(echo $COUNTS | awk '{print $1+$2+$3}')

        if [ $TOTAL_COUNT -eq $LAST_COUNT ] && [ $TOTAL_COUNT -gt 0 ]; then
            STALE_COUNT=$((STALE_COUNT + 1))
            log "WARNING: No progress in last check (stale count: $STALE_COUNT/3). Counts: $COUNTS"

            if [ $STALE_COUNT -ge 3 ]; then
                log "ERROR: Pipeline appears stalled! Restarting..."
                kill $(cat $PID_FILE) 2>/dev/null
                sleep 10
                # Kill any orphaned workers
                pkill -f "worker_process" 2>/dev/null
                sleep 5
                start_pipeline
                STALE_COUNT=0
            fi
        else
            STALE_COUNT=0
            log "Pipeline running OK. Progress - AIGI: $(echo $COUNTS | cut -d' ' -f1), eval24: $(echo $COUNTS | cut -d' ' -f2), ReWIND: $(echo $COUNTS | cut -d' ' -f3)"
        fi

        LAST_COUNT=$TOTAL_COUNT
    else
        # Pipeline not running
        if [ -f "$SCRIPT_DIR/Results/final_report.txt" ]; then
            log "Pipeline completed (final report exists)"
            break
        fi

        log "Pipeline not running - restarting..."
        sleep 5
        # Kill any orphaned workers
        pkill -f "worker_process" 2>/dev/null
        sleep 5
        start_pipeline
        STALE_COUNT=0
    fi

    sleep 300  # Check every 5 minutes
done

log "Watchdog exiting"
