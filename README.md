# Challenges of Pixel Art Generation with GANs

This repository contains the source code used by the paper _"On the Challenges of Pixel Art Generation with GANs"_ 
to be presented at AIIDE 2022.


## Installation

The code uses **Tensorflow** 2.9.1 and requires the following dependencies to be installed:

- jupyter
- tensorflow
- tensorflow_addons
- matplotlib
- scipy
- scikit-image

For convenience, a [requirements.txt](requirements.txt) file is available with the necessary packages 
and their versions. In a fresh virtual environment, you can:

```shell
pip install -r requirements.txt
```


## Running

The notebook [experiments.ipynb](experiments.ipynb) can be executed from the first to the last cell to execute
the training procedure of the four models presented in the paper:

1. baseline (no aug.)
2. baseline
3. palette-indexed
4. with histogram loss

To define which model to train, you can set the value of the model variable in the 3rd cell:
```py
MODELS = ["baseline (no aug.)", "baseline", "indexed", "histogram"]
model = MODELS[0]  # <-- CHOOSE HERE
```

There is also some configurations over [configuration.py](configuration.py), like the batch size, the maximum size 
of a color palette and so on.

Tensorboard can be used to debug the performance of the models during training. You can point its log directory to
`temp-side2side` to see the logged scalars and images:

```shell
tensorboard --logdir temp-side2side
```