#!/bin/bash

echo "experiment 3, jump, is_over_all_best"
for RUN in 1 2 3 4 5
do
  echo "------------------------------------------  Run $RUN ------------------------------------------"
  python3 main_experiment_3.py --experiment '3' --run $RUN --train_file_path '../SCAN/add_prim_split/tasks_train_addprim_jump.txt' --test_file_path '../SCAN/add_prim_split/tasks_test_addprim_jump.txt' --is_train True --is_over_all_best True --add_prim 'turn_left'
done
