#!/bin/bash

# Define log files
LOG_FILE="logfile.log"
BACKEND_LOG="backend.log"

echo "$(date) - Checking models/model..." >> "$LOG_FILE"

# Check if models/model directory is empty
if [ -z "$(ls -A models/model)" ]; then
    echo "$(date) - models/model is empty. Running dvc repro..." >> "$LOG_FILE"
    dvc repro >> "$LOG_FILE" 2>&1
    
    echo "$(date) - Starting backend server..." >> "$LOG_FILE"
    # Kill any existing backend server just in case
    pkill -f "python src/backend.py"
    
    # Start backend
    nohup python src/backend.py >> "$BACKEND_LOG" 2>&1 &
else
    echo "$(date) - models/model is not empty. Running fine-tuning..." >> "$LOG_FILE"
    
    # Run finetune.py
    python src/finetune.py >> "$LOG_FILE" 2>&1
    
    echo "$(date) - Restarting backend server..." >> "$LOG_FILE"
    # Kill existing backend server
    pkill -f "python src/backend.py"
    
    # Start backend fresh
    nohup python src/backend.py >> "$BACKEND_LOG" 2>&1 &
fi
