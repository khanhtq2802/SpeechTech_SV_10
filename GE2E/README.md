# PyTorch Implementation of "Generalized End-to-End Loss for Speaker Verification"

## Introduction

This repository contains my implementation of Generalized End-to-End Loss for Speaker Verification (ICASSP 2019) in PyTorch.

## Requirements

To install the python dependencies for this project, run

```
pip install -r requirements.txt
```

### Note

If `pip` defaults to `pip2` in your system, replace it with `pip3`.

## Usage

Run preprocessing data

```
python main.py --preprocess
```

To train the Embedder and Loss Module run

```
python main.py --train
```

To run evaluation to produce the Equal-Error-Rate and threshold at which the EER is achieved, run

```
python main.py --eval
```

To calculate the similarity score between two audio files (decide whether they have the same speaker), run

```
python main.py --similarity --path1 <path-to-audio-1> --path2 <path-to-audio-2>
```

### Note

This work was originally done using a Kaggle notebook.

## References

[1] "Generalized End-to-End Loss for Speaker Verification", Wan et. al., ICASSP 2018

[2] https://github.com/HarryVolek/PyTorch_Speaker_Verification
