# ASVspoof 2019 LA Anti-Spoofing

PyTorch implementation of a spoofing detection system for the ASVspoof 2019 Logical Access (LA) dataset.

## Task

The goal is to classify an audio recording as either:

- **bonafide** — genuine speech;
- **spoof** — synthesized or converted speech.

The quality of the model is evaluated using the **Equal Error Rate (EER)** metric.

## Model

The project uses:

- LFCC feature extraction;
- weighted CrossEntropyLoss;
- Adam optimizer;
- StepLR scheduler.

## Dataset

Experiments were conducted on the **ASVspoof 2019 LA** dataset.

Dataset splits:

- Train
- Development (validation)
- Evaluation (test)

## Project structure

```
src/
    configs/
    datasets/
    loss/
    metrics/
    model/
    trainer/
train.py
inference.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Inference

```bash
python inference.py
```

## Results

Final evaluation:

- EER: **3.47%**

## Acknowledgements

The project is based on the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template),
which was adapted for the ASVspoof 2019 LA anti-spoofing task.