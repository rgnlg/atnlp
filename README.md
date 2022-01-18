# Advanced Topics in Natural Language Processing (ATNLP)

The project is done by Duo Yang, Olga Iarygina, and Philine Zeinert as a part of the course Advanced Topics in Natural Language Processing at the University of Copenhagen
Dec 2021 - Jan 2022

### Paper
The aim of the project was to reimplement the paper "Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks" (Lake & Baroni, 2018) 

Paper: https://arxiv.org/pdf/1711.00350.pdf <br />
Data: https://github.com/brendenlake/SCAN <br />
Tutorial: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html <br />

Lake, B.M., & Baroni, M. (2018). Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. ICML.

### Requirements
- numpy
- random
- torch
- __future__
- io
- os
- unicodedata
- string
- re
- argparse
- pathlib

### Execution

- download files from https://github.com/brendenlake/SCAN and store them with a given tree structure in the directory "SCAN" on the same level where run.sh and the python files are
- for each of the experiments configurations, parameter can be adjusted and specified in "configurations.json", e.g.

  ```
  {
    "experiment": '1',
    "train_file_path":'../SCAN/add_prim_split/tasks_train_addprim_jump.txt',
    "test_file_path": '../SCAN/add_prim_split/tasks_test_addprim_jump.txt',
    "is_train": true,
    "is_over_all_best": true,
    "prim": false
  }
  ```
where:
  - "experiment" - experiment to be executed;
  - "train_file_path" - path to the training data;
  - "test_file_path" - path to the test data;
  - "is_train" - training process;
  - "is_over_all_best" - model to be executed (the overall best model or the best model for a particular experiment);
  - "prim" - exposion of primitive commands only denoting a certain basic action.
  
  - Experiments can be executed using ./run.sh (Note: the shell-file might be configured as "executable" to be able to run it, using chmod +x run.sh)

### Structure
- `dataloader.py` - loading data files;
- `model.py` - Seq2Seq models;
- `train.py` - training;
- `evaluate.py` - evaluation;
- `main.py` - execution of the experiments;
- `configurations.json` - experiments' configurations;
- `run.sh` - shell file to execurte experiments.


