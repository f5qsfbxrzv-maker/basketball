@echo off
cd /d "c:\Users\d76do\OneDrive\Documents\New Basketball Model"
python scripts\optuna_tune_25features.py > tuning_log.txt 2>&1
echo Tuning complete! Check tuning_log.txt and models/ folder
pause
