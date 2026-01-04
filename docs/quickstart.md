# Quickstart

## Train

```bash
python scripts/train.py --config configs/experiment/main_experiment.yaml --data_dir ./data
```

## Evaluate

```bash
python scripts/evaluate.py --config configs/experiment/main_experiment.yaml   --checkpoint outputs/checkpoints/best.ckpt --data_dir ./data
```

## Predict

```bash
python scripts/predict.py --config configs/experiment/main_experiment.yaml   --checkpoint outputs/checkpoints/best.ckpt --image path/to/byteplot.png
```
