#!/bin/bash

echo "Re-Producing Lake & Baroni (2018) Experiments - Group 6"
for RUN in 1 2 3 4 5
do
  echo "------------------------------------------  Run $RUN ------------------------------------------"
  python3 main.py --configurations_path 'configurations.json' --run $RUN
done