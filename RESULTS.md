# Reported Results

This document summarizes the quantitative values stated in the supplied DermFusion manuscript. It is not a new experiment.

## HAM10000 test fold

| Metric | Value |
|---|---:|
| Accuracy | 87.82% |
| 95% CI | 86.9–88.6% |
| Balanced Accuracy | 87.77% |
| Macro F1 | 0.8497 |
| ROC-AUC | 0.9799 |
| Macro PR-AUC | 0.9172 |
| MEL sensitivity | 88.80% |
| MEL sensitivity, adjusted threshold | 91.20% |
| ECE after calibration | 0.1575 |
| Uncalibrated ECE | 0.4395 |
| Segmentation Dice | 0.9359 |
| Segmentation IoU | 0.8880 |

## External validation

| Dataset | Accuracy | AUC | F1 | Dice |
|---|---:|---:|---:|---:|
| PH2 | 86.4% | 0.971 | 0.832 | 0.918 |
| ISIC 2019 | 84.1% | 0.958 | 0.811 | 0.894 |
| PAD-UFES-20 | 82.5% | 0.942 | 0.796 | 0.871 |

No external fine-tuning is reported for these evaluations.

## Ablation

| Configuration | Accuracy | AUC | Dice | ECE |
|---|---:|---:|---:|---:|
| Full DermFusion | 87.82% | 0.9799 | 0.9359 | 0.1575 |
| Without Cross-Attention | 85.90% | 0.9602 | 0.8872 | 0.214 |
| Without Segmentation Branch | 86.40% | 0.9654 | N/A | 0.208 |
| Without Deep Supervision | 86.72% | 0.9711 | 0.9142 | 0.194 |
| Without MC Dropout | 87.11% | 0.9750 | 0.9330 | 0.241 |
| Without Temperature Calibration | 87.20% | 0.9798 | 0.9351 | 0.312 |
| Without Uncertainty Weighting | 86.84% | 0.9719 | 0.9210 | 0.226 |

## Cross-validation stability reported in the manuscript

| Fold | Accuracy | AUC | Dice | F1 |
|---|---:|---:|---:|---:|
| 1 | 87.9% | 0.978 | 0.921 | 0.846 |
| 2 | 88.6% | 0.982 | 0.927 | 0.854 |
| 3 | 89.4% | 0.985 | 0.933 | 0.862 |
| 4 | 88.1% | 0.979 | 0.925 | 0.849 |
| 5 | 88.7% | 0.983 | 0.929 | 0.858 |
| Mean ± SD | 88.5 ± 0.7% | 0.981 ± 0.003 | 0.927 ± 0.005 | 0.854 ± 0.007 |

## Retrospective uncertainty triage

At the reported referral threshold `τ*=0.30`:

- Unfiltered accuracy: 87.82%.
- Retained autonomous coverage: 68%.
- Retained accuracy: 96.40%.
- Retained MEL sensitivity: 97.10%.
- Referred cohort: 32%.

These are retrospective held-out test-fold simulations, not prospective clinical validation.

## Computational complexity

The manuscript reports the following benchmark values at 384×384 input, batch size 1, on NVIDIA Tesla T4:

| Method | Params (M) | FLOPs (G) | Deterministic latency (ms) | MC latency (ms) | GPU memory (GB) |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 25.6 | 8.2 | 20–28 | – | 0.4 |
| EfficientNet | 19.3 | 9.5 | 25–35 | – | 0.5 |
| ViT-B/16 | 86.6 | 55.4 | 45–65 | – | 1.2 |
| Swin-Unet | 108.0 | 62.1 | 55–75 | – | 1.8 |
| MedMamba | 28.4 | 14.2 | 30–40 | – | 0.6 |
| DermFusion | 124.3 | 98.7 | 85–120 | 850–1200 | 3.2 |

The manuscript notes that reducing MC dropout passes from 10 to 3 lowers latency to approximately 280 ms with an ECE increase of less than 0.003.

## Important distinction

The supplied training log contains only an excerpt through epoch 21. For example, epoch 19 reports accuracy 0.9111 and AUC 0.9877, while the manuscript's final held-out test-fold table reports 87.82% accuracy and 0.9799 AUC. These should not be treated as the same evaluation point.

See `REPRODUCIBILITY_NOTES.md` for source-to-code discrepancies that should be resolved before claiming exact reproduction.
