# Representation Learning (CIFAR-10)

Compare small CNNs on CIFAR-10 with a simple trainer and optional logging.

## Models
- LeNetSmall (lenet)
- VGGTiny (vgg)
- ResNet18Tiny (resnet)
- MobileNetV2Tiny (mobilenet)

## Quick Start
```bash
python scripts/rep_train.py --config src/representation_learning/exp/configs/cifar10_resnet.yaml
python scripts/rep_infer.py --config src/representation_learning/exp/configs/cifar10_resnet.yaml --ckpt models/representations/exp/cifar10_resnet/last.pt --sample-index 3
```

## Config Notes
- Data at ./data is reused
- Use max_train/max_test for subsets
- Logging via misc: logging, csv_logging, log_confusion, log_feature_maps, log_gradcam, log_every_n
- Checkpoints: models/representations/exp/<experiment_name>/

## Tips
- On macOS (MPS), set num_workers: 0
- Start with resnet for best subset accuracy