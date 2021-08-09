# Soft-Sensing Graph Neural Network for Multivariate Time-series Classification

This repository is the Pytorch implementation of Soft Sensing Graph Neural Network for
Multivariate Time-series Classfification.

## Requirements

Recommended version of OS & Python:

* **OS**: Ubuntu 18.04.2 LTS
* **Python**: python3.7 ([instructions to install python3.7](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/)).

To install python dependencies, virtualenv is recommended, `sudo apt install python3.7-venv` to install virtualenv for python3.7. All the python dependencies are verified for `pip==20.1.1` and `setuptools==41.2.0`. Run the following commands to create a venv and install python dependencies:

```setup
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Required python packages:

```install
import os
import math
import time
import numpy as np
import pandas as pd
import torch
import sklearn
import scipy
import warnings
```

## Datasets

[Seagate soft sensing data sets](https://github.com/Seagate/softsensing_data),


We can get the raw data through the links above. The raw data is under the folder `BigDataChallenge/data//` of the above github link.

## Training and Evaluation

The training procedure and evaluation procedure are all included in the `Soft_sensing_GNN.py`.
To evaluate on dataset, run the following command:

```evaluate 
predictions, targets = inference(model, data_loader, adj, device)
plot_probas(predictions, targets, label_cols, softmax=False, rescale=False, reverse=False)
```

The detailed descriptions about the hyper-parameters are as following:

| Parameter name | Description of parameter |
| --- | --- |
| time_step | length of time steps in each sample, default 2 |
| epoch | epoch size during training |
| lr | learning rate |
| multi_layer | hyper parameter of GNN which controls the parameter number of hidden layers, default 4 |
| device | device that the code works on, 'cpu' or 'cuda' | 
| validate_freq | frequency of validation, default 1 |
| batch_size | batch size, default 1024|
| weight_decay | method for adding a L2 penalty to the cost , default 1e-4 |
| early_stop | the patience to enable early stop, default 50 |


**Table 1** Configurations for NXD datasets
| Labels | train-positives | train-negatives | valid-positives | valid-negatives | test-positives | test-negatives |
| -----   | ---- | ---- |---- |---- |---- | --- |
| MEAS2_SRHO | 6020 | 272| 1417 | 13 | 878 | 10 |
| MEAS_LOOPER | 10288 | 33 | 1509 | 5 | 950 | 2 |
| MEAS_METAPULSE | 42989 | 200 | 7795 | 43 | 5414 | 48 |
| MEAS_MOKEM | 11114 | 132 | 1594 | 23 | 1989 | 33 |
| MEAS_MON_LOOPER | 32794 | 428 | 4283 | 91 | 3567 | 49 |
| MEAS_MON_SRHO | 64007 | 709 | 11833 | 68 | 9123 | 86 |
| MEAS_MON_XRF| 117332 | 1702 | 19663 | 482 | 16975 | 371 |
| MEAS_MON_XRR| 1748 | 443 | 196 | 39 | 975 | 8 |
| MEAS_SRHO| 22420 | 86 | 4225 | 6 | 2906 | 12 |
| MEAS_THCK| 7874 | 48 | 1788 | 4 | 1151 | 5 | 
| MEAS_XRF| 35874 | 227 | 6231 | 36 | 5114 | 43 |

