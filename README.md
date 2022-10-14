## Optimizing Neural Networks with Gradient Lexicase Selection

[Optimizing Neural Networks with Gradient Lexicase Selection](https://openreview.net/forum?id=J_2xNmVcY4), Ding & Spector, ICLR 2022.

---
Basic usage:
- To train and evaluate baseline architectures, do
```
python3 base.py
```

- To train and evaluate gradient lexicase selection, do
```
python3 lexi.py
```

Also use the `--help` flag to see instructions for optional arguments to configure architecture and dataset:

- The architectures are indexed as follows (from 0 to 6):
VGG, ResNet18, ResNet50, DenseNet121, MobileNetV2, SENet18, EfficientNetB0

- The datasets are specified as follows:
'C10' - CIFAR-10, 'C100' - CIFAR-100, 'SVHN' - SVHN

Please contact the authors for further questions.
