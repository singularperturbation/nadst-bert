# NADST-BERT (WIP)

**NOTE**: This is a work in progress and will update later when I have time.
Will be figuring out DST dataset format as I go as well, which will slow down
progress.

Trying to do a TF2 implementation of the
[Non-Autoregressive Dialog State Tracking](https://arxiv.org/abs/2002.08024)
paper.

I'm using the [PyTorch implementation](https://github.com/henryhungle/NADST/) as
a reference, and plan to train on Colab on the MultiWoz 2.2
[dataset](https://huggingface.co/datasets/multi_woz_v22).

I'll use a pretrained BERT model for the feature engineering layer.


## Licensing

The original implementation of NADST is MIT-licensed and Copyright (c) 2020 Hung
Le. The
[positional encoding](https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb)
implementation I use is originally from Tensorflow Text, and is Copyright (c)
2019 The Tensorflow Authors and licensed under the Apache License, Version 2.0.

This implementation is MIT-licensed except for the positional encoding, which is
licensed under Apache V2.
