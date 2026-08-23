DermFusion
DermFusion: An Uncertainty-Aware Cross-Attention CNN–Transformer Framework for Multi-Task Skin Lesion Classification and Segmentation

This repository packages the research paper, training implementation, experiment log excerpt, and reproducibility notes for DermFusion.

Research / decision-support software. This repository is not a medical device and must not be used as an autonomous diagnostic system.

Contents
DermFusion-GitHub/
├── README.md
├── CITATION.cff
├── REPRODUCIBILITY_NOTES.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── dermfusion_train.py
├── results/
│   └── training_log_excerpt.md

└── docs/
    └── RESULTS.md
Method
DermFusion combines:

ResNet-50 with sequential CBAM refinement for local texture and morphology.
ViT-B/16 at 384×384 resolution, producing 576 patch tokens.
768-dimensional projection and bidirectional CNN↔ViT cross-attention.
Morphology-guided attention and a morphology-conditioned feature gate.
A U-Net-style segmentation decoder with deep supervision.
Focal classification loss and Focal-Tversky segmentation loss.
Learnable homoscedastic uncertainty weighting.
Monte Carlo dropout for predictive uncertainty.
Post-hoc temperature scaling.
Uncertainty-guided selective prediction / referral.
The uploaded implementation uses 7 HAM10000 classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC.

DermFusion architecture

Dataset
The paper reports experiments on HAM10000 (10,015 images, seven classes) and uses lesion segmentation masks associated with the ISIC 2018 Task 1 annotation set.

The repository does not redistribute any dataset or masks. See data/README.md.

Hardware / training
The paper reports training on a single NVIDIA Tesla T4 with 15.6 GB VRAM. The supplied implementation is configured for 384×384 input, 45 epochs, mixed precision, AdamW, cosine scheduling, EMA stabilization, and checkpoint/resume support.

Run
Install dependencies:

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
Set the dataset directory:

export DERMFUSION_DATA_DIR=/path/to/ham1000-segmentation-and-classification
export DERMFUSION_CKPT_PATH=./checkpoints/dermfusion_checkpoint.pth
export DERMFUSION_BEST_PATH=./checkpoints/dermfusion_best.pth
python src/dermfusion_train.py
On Windows PowerShell:

$env:DERMFUSION_DATA_DIR="C:\path\to\ham1000-segmentation-and-classification"
$env:DERMFUSION_CKPT_PATH=".\checkpoints\dermfusion_checkpoint.pth"
$env:DERMFUSION_BEST_PATH=".\checkpoints\dermfusion_best.pth"
python src\dermfusion_train.py
The default path remains compatible with the original Kaggle environment if the environment variable is not set.

Reported manuscript results
The supplied manuscript reports the following principal HAM10000 test-fold results:

Metric	Reported value
Accuracy	87.82%
Balanced Accuracy	87.77%
Macro F1	0.8497
ROC-AUC	0.9799
Macro PR-AUC	0.9172
MEL sensitivity	88.80%
MEL sensitivity, adjusted threshold	91.20%
Segmentation Dice	0.9359
Segmentation IoU	0.8880
Calibrated ECE	0.1575
Optimal calibration temperature	1.75
Referral threshold	0.30
Retained coverage	68%
Retained accuracy	96.4%
Retained MEL sensitivity	97.1%
The manuscript also reports external no-fine-tuning evaluation on PH2, ISIC 2019, and PAD-UFES-20, plus ablation, calibration, fairness, cross-validation, and failure-mode analyses. See docs/RESULTS.md and the PDF.

Important reproducibility note
The repository deliberately preserves the uploaded implementation and does not silently rewrite experimental claims.

There are several source-to-code consistency items that should be resolved before claiming that the repository exactly reproduces every number in the manuscript:

The manuscript describes a strict patient-wise split using lesion_id, whereas the supplied training code currently calls StratifiedKFold directly on image rows.
The manuscript reports 124.3M parameters, while the supplied training log reports 137.0M parameters.
The code configuration uses an effective batch size of 8 × 4 = 32, while the manuscript reports an effective batch size of 16.
The code configuration sets classification focal-loss gamma=1.5, while the code header describes gamma=2.
The supplied training log is an excerpt through epoch 21, while the manuscript reports final test-fold results after the full experimental workflow.
The manuscript states that patient-wise split indices were publicly provided, but those split-index files were not among the uploaded materials used to build this package.
These are documented rather than hidden so that a public GitHub release remains scientifically auditable.

Limitations
The manuscript explicitly notes that:

baseline models were not all retrained under identical conditions;
the uncertainty/referral analysis is retrospective and is not prospective clinical validation;
training data originate from a limited acquisition setting;
diverse skin phototypes, particularly Fitzpatrick V–VI, require additional evaluation;
prospective multi-centre validation is required before clinical deployment.
Citation
See CITATION.cff for a machine-readable citation record.

License / reuse
No open-source software license has been assumed in this package. Before making the repository public, the authors should choose and add an appropriate license covering the code and separately verify the redistribution terms of the paper, datasets, pretrained weights, and third-party components.

Contact
See the author and corresponding-author information in the included paper.
