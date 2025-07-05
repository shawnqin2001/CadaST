#! /bin/bash
source ~/.bashrc
micromamba activate cadast
target="ablation.py"
python  $target

if [ $? -eq 0 ]; then
    echo "Python script executed successfully."
    curl -d "Task $target is done" ntfy.sh/shawn_alert
else
    echo "Python script failed with exit code $?."
    curl -d "Task $target failed" ntfy.sh/shawn_alert
fi
