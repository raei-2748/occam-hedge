#!/bin/bash
# Monitor the production run and execute post-analysis when finished.

LOG_FILE="production_run.log"
SCRIPT_NAME="run_paper.py"
POST_ANALYSIS_SCRIPT="scripts/run_post_analysis.sh"

echo "[Monitor] Watching for $SCRIPT_NAME..."

while pgrep -f "$SCRIPT_NAME" > /dev/null; do
    sleep 30
done

echo "[Monitor] $SCRIPT_NAME has finished."

# Check for success in log (primitive check)
if grep -q "Paper Run Complete" "$LOG_FILE" || grep -q "Traceback" "$LOG_FILE"; then
    if grep -q "Traceback" "$LOG_FILE"; then
        echo "[Monitor] WARNING: Traceback detected in log. Run might have failed."
    else
        echo "[Monitor] Run appears successful."
    fi
else
    echo "[Monitor] Process exited but 'Paper Run Completed' not found. Check logs."
fi

# Extract Run Directory
RUN_DIR=$(grep "Directory:" "$LOG_FILE" | tail -n 1 | awk '{print $2}')

if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
    echo "[Monitor] Detected Run Directory: $RUN_DIR"
    echo "[Monitor] Starting Post-Analysis..."
    
    # Run Post-Analysis
    bash "$POST_ANALYSIS_SCRIPT" "$RUN_DIR" >> "$RUN_DIR/post_analysis.log" 2>&1
    
    echo "[Monitor] Post-Analysis complete. See $RUN_DIR/post_analysis.log"
else
    echo "[Monitor] Could not determine Run Directory from $LOG_FILE. Skipping post-analysis."
fi
