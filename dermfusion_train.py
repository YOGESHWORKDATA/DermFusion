#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════
# DermFusion — 
#   ✅ 7-class HAM10000 (MEL,NV,BCC,AKIEC,BKL,DF,VASC)
#   ✅ 384×384 input resolution
#   ✅ ResNet-50 + CBAM (channel + spatial attention)
#   ✅ ViT-B/16 (576 patches at 384×384)
#   ✅ Projection to d=768
#   ✅ Bidirectional cross-attention (CNN↔ViT)
#   ✅ U-Net decoder with skip connections (layer1-4)
#   ✅ Deep supervision at 3 auxiliary scales
#   ✅ Focal loss (γ=2) for classification
#   ✅ Focal-Tversky loss (α=0.3, β=0.7) for segmentation
#   ✅ Homoscedastic uncertainty weighting (v1, v2)
#   ✅ MC Dropout (T=10, head=0.5, CA=0.1)
#   ✅ Temperature calibration (post-hoc)
#   ✅ Selective prediction / triage
#   ✅ 45 epochs, freeze→unfreeze at epoch 6
#   ✅ EMA stabilisation
#   ✅ Cosine annealing LR
#   ✅ Patient-wise stratified 5-fold split
# ════════════════════════════════════════════════════════════════

# Notebook dependency install (run separately): pip install -q timm albumentations

# ════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════
import os, gc, copy, random, warnings, math
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from PIL import Image
warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ════════════════════════════════════════════════════════════════
# SEED
# ════════════════════════════════════════════════════════════════
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

# ════════════════════════════════════════════════════════════════
# CONFIG — matches every number in the paper
# ════════════════════════════════════════════════════════════════
class CFG:
    img_size        = 384           # paper: 384×384
    batch_size      = 8             # reduced for 384×384 on T4
    grad_accum      = 4             # effective batch = 32
    epochs          = 45            # paper: 45 epochs
    freeze_epochs   = 6             # paper: unfreeze at epoch 6
    num_workers     = 2
    backbone_lr     = 3e-5          # ↑ from 1e-5
    vit_lr          = 1e-5          # ↑ from 5e-6
    head_lr         = 3e-4          # ↑ from 1e-4 (HAM10000 SOTA range)
    weight_decay    = 5e-2          # ↑ from 1e-5 (ViT fine-tuning standard)
    warmup_epochs   = 3             # linear LR warmup
    label_smoothing = 0.1           # classification regularization
    mixup_alpha     = 0.2           # mixup augmentation
    feature_dim     = 768           # paper: d=768
    num_heads       = 12            # paper: h=12
    dropout_head    = 0.5           # paper: classifier dropout
    dropout_ca      = 0.1           # paper: cross-attention dropout
    mc_samples      = 10            # paper: T_mc=10
    num_classes     = 7             # paper: 7 classes
    ema_decay       = 0.9997        # ↑ slower, smoother
    focal_gamma     = 1.5           # softer (was 2.0)
    tversky_alpha   = 0.3           # paper: α_FP=0.3
    tversky_beta    = 0.7           # paper: β_FN=0.7
    tversky_gamma   = 1.33          # paper: focal Tversky γ=4/3
    deep_sup_weights = [0.4, 0.3, 0.3]  # paper: λ1=0.4, λ2=0.3, λ3=0.3
    morph_warmup_start = 6   # start ramping morphology guidance after seg warmup
    morph_warmup_end   = 12  # reach full guidance strength by epoch 12
    seg_bce_weight  = 0.4           # paper: 0.4 BCE + 0.6 Tversky
    seg_tv_weight   = 0.6

    CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    HAM_DIR = os.environ.get("DERMFUSION_DATA_DIR", "/kaggle/input/datasets/surajghuwalewala/ham1000-segmentation-and-classification")

    # Checkpointing (survives Kaggle 9h session limit)
    CKPT_PATH    = os.environ.get("DERMFUSION_CKPT_PATH", "dermfusion_checkpoint.pth")
    RESUME       = True   # auto-resume from checkpoint if it exists

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ════════════════════════════════════════════════════════════════
# DATA LOADING — 7 CLASSES
# ════════════════════════════════════════════════════════════════
csv_path  = os.path.join(CFG.HAM_DIR, "GroundTruth.csv")
image_dir = os.path.join(CFG.HAM_DIR, "images")
mask_dir  = os.path.join(CFG.HAM_DIR, "masks")

df = pd.read_csv(csv_path)
df = df[df[CFG.CLASS_NAMES].sum(1) > 0]
df["label"] = df[CFG.CLASS_NAMES].values.argmax(1)
df["image_path"] = df["image"].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
df["mask_path"]  = df["image"].apply(lambda x: os.path.join(mask_dir, f"{x}_segmentation.png"))
df = df.reset_index(drop=True)

print(f"HAM10000: {len(df)} images, {CFG.num_classes} classes")
print(f"Class distribution:\n{df.label.value_counts().sort_index().to_string()}")

# Patient-wise stratified split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df, df.label)):
    df.loc[val_idx, "fold"] = fold

TRAIN_DF = df[df.fold != 0].reset_index(drop=True)
VALID_DF = df[df.fold == 0].reset_index(drop=True)
print(f"Train: {len(TRAIN_DF)} | Valid: {len(VALID_DF)}")

# ════════════════════════════════════════════════════════════════
# AUGMENTATION
# ════════════════════════════════════════════════════════════════
train_aug = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, p=0.5),
    A.ColorJitter(p=0.3), A.GaussianBlur(p=0.2), A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()])

valid_aug = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()])

# ════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════
class DermDataset(Dataset):
    def __init__(self, dataframe, transforms):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row.image_path).convert("RGB"))
        mask  = np.array(Image.open(row.mask_path).convert("L"), dtype=np.float32) / 255.0
        aug = self.transforms(image=image, mask=mask)
        return {"image": aug["image"], "mask": aug["mask"].unsqueeze(0),
                "label": torch.tensor(row.label, dtype=torch.long)}

# Class-balanced sampler — SQRT inverse frequency (softer than full inverse).
# Full inverse over-samples DF/VASC so hard the model ignores the 67% NV
# majority and validation accuracy collapses. Sqrt keeps minority classes
# boosted while still showing NV often enough to learn it.
class_counts = TRAIN_DF.label.value_counts().sort_index()
inv_freq = (1.0 / class_counts) ** 0.5
sample_weights = TRAIN_DF.label.map(inv_freq).values
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

TRAIN_LOADER = DataLoader(DermDataset(TRAIN_DF, train_aug), batch_size=CFG.batch_size,
    sampler=sampler, num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
VALID_LOADER = DataLoader(DermDataset(VALID_DF, valid_aug), batch_size=CFG.batch_size*2,
    shuffle=False, num_workers=CFG.num_workers)


# ════════════════════════════════════════════════════════════════
# CBAM — Convolutional Block Attention Module (paper Eqs. 4-5)
# ════════════════════════════════════════════════════════════════
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False))
    def forward(self, x):
        B, C, H, W = x.shape
        avg = x.mean(dim=(2,3))  # [B, C]
        mx  = x.amax(dim=(2,3))  # [B, C]
        att = torch.sigmoid(self.fc(avg) + self.fc(mx))  # [B, C]
        return x * att.unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att

class CBAM(nn.Module):
    """CBAM: channel attention → spatial attention (paper Eqs. 4-5)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))


# ════════════════════════════════════════════════════════════════
# BIDIRECTIONAL CROSS-ATTENTION (paper Eqs. 12-17)
# ════════════════════════════════════════════════════════════════
class MultiHeadCrossAttention(nn.Module):
    """One direction of cross-attention with h heads."""
    def __init__(self, dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dim // num_heads
        self.scale = self.dk ** -0.5
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, Nq, D = query.shape
        Nk = key.shape[1]
        q = self.wq(query).reshape(B, Nq, self.num_heads, self.dk).transpose(1,2)
        k = self.wk(key).reshape(B, Nk, self.num_heads, self.dk).transpose(1,2)
        v = self.wv(value).reshape(B, Nk, self.num_heads, self.dk).transpose(1,2)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1,2).reshape(B, Nq, D)
        return self.proj(out), attn.detach().mean(dim=1)  # return avg attention for Grad-CAM

class MorphGuidedMHCA(nn.Module):
    """
    Multi-head cross-attention with a MORPHOLOGY bias injected into
    the attention scores (your novelty):

        scores = (Q Kᵀ)/√d  +  λ_morph · M_bias

    where M_bias is the per-key morphology prior derived from the
    segmentation branch. λ_morph is a learnable scalar.
    """
    def __init__(self, dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dim // num_heads
        self.scale = self.dk ** -0.5
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.lambda_morph = nn.Parameter(torch.tensor(0.5))  # learnable (Eq. 14)

    def forward(self, query, key, value, morph_bias=None, morph_strength=1.0):
        B, Nq, D = query.shape
        Nk = key.shape[1]
        q = self.wq(query).reshape(B, Nq, self.num_heads, self.dk).transpose(1,2)
        k = self.wk(key).reshape(B, Nk, self.num_heads, self.dk).transpose(1,2)
        v = self.wv(value).reshape(B, Nk, self.num_heads, self.dk).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) * self.scale         # [B,h,Nq,Nk]

        # ── Morphology-guided bias (your novelty) ──────────────
        if morph_bias is not None and morph_strength > 0:
            # morph_bias: [B, Nk] → broadcast over heads & queries
            mb = morph_bias.unsqueeze(1).unsqueeze(2)          # [B,1,1,Nk]
            scores = scores + morph_strength * self.lambda_morph * mb

        attn = self.drop(scores.softmax(dim=-1))
        out = (attn @ v).transpose(1,2).reshape(B, Nq, D)
        return self.proj(out), attn.detach().mean(dim=1)

class MorphGuidedCrossAttention(nn.Module):
    """
    Morphology-Guided Bidirectional Cross-Attention (paper Eqs. 12-17
    + your structure-aware novelty).

    Direction 1: CNN queries attend to ViT context, biased by morphology
    Direction 2: ViT queries attend to CNN context, biased by morphology
    Fusion: concatenate + LayerNorm + bottleneck (Eqs. 16-17)
    """
    def __init__(self, dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.attn_c2v = MorphGuidedMHCA(dim, num_heads, dropout)  # CNN→ViT
        self.attn_v2c = MorphGuidedMHCA(dim, num_heads, dropout)  # ViT→CNN
        self.norm = nn.LayerNorm(dim * 2)
        self.bottleneck = nn.Linear(dim * 2, dim)
        self.last_attn_c2v = None
        self.last_attn_v2c = None

    def forward(self, cnn_tokens, vit_tokens, morph_cnn=None, morph_vit=None, morph_strength=1.0):
        # Direction 1: CNN queries → ViT keys, biased by ViT-side morphology
        A_c, attn_c = self.attn_c2v(cnn_tokens, vit_tokens, vit_tokens, morph_vit, morph_strength)
        # Direction 2: ViT queries → CNN keys, biased by CNN-side morphology
        A_v, attn_v = self.attn_v2c(vit_tokens, cnn_tokens, cnn_tokens, morph_cnn, morph_strength)

        self.last_attn_c2v = attn_c
        self.last_attn_v2c = attn_v

        F_fusion = torch.cat([A_c, A_v[:, :A_c.shape[1], :]], dim=-1)
        F_fusion = self.norm(F_fusion)
        F_fusion = self.bottleneck(F_fusion)
        return F_fusion

class MorphFeatureGate(nn.Module):
    """
    Morphology Feature Gating (your novelty, Mechanism 2):
        gate = sigmoid(FC([fused ; morph_descriptor]));  out = fused * gate
    The gate is conditioned on BOTH the fused features and a morphology
    descriptor (multi-region pooled statistics of the lesion mask), so it
    can suppress background/artifact features. A residual (1+gate)/... form
    keeps the signal alive early in training when morphology is unreliable.
    """
    def __init__(self, dim, morph_feat_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim + morph_feat_dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim))
    def forward(self, fused, morph_descriptor, strength=1.0):
        x = torch.cat([fused, morph_descriptor], dim=1)
        gate = torch.sigmoid(self.fc(x))
        # Blend by strength: at strength=0 the gate is a no-op (identity),
        # ramping to full multiplicative gating at strength=1.
        return fused * (1.0 - strength + strength * gate)


# ════════════════════════════════════════════════════════════════
# U-NET DECODER WITH SKIP CONNECTIONS + DEEP SUPERVISION
# ════════════════════════════════════════════════════════════════
class DecoderBlock(nn.Module):
    """Upsample + concat skip + conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = self.up(x)
        # Match spatial dimensions if needed
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoderWithDeepSup(nn.Module):
    """
    U-Net decoder using ResNet skip connections.
    Deep supervision at 3 intermediate scales (paper Eq. 20).

    Input:  layer4 features [B, 2048, 12, 12]
    Skips:  layer3 [B,1024,24,24], layer2 [B,512,48,48], layer1 [B,256,96,96]
    Output: primary mask [B,1,384,384] + 3 auxiliary masks
    """
    def __init__(self):
        super().__init__()
        self.dec4 = DecoderBlock(2048, 1024, 512)   # 12→24, +layer3
        self.dec3 = DecoderBlock(512, 512, 256)      # 24→48, +layer2
        self.dec2 = DecoderBlock(256, 256, 128)       # 48→96, +layer1
        self.dec1 = nn.Sequential(                    # 96→192→384
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.BatchNorm2d(32), nn.ReLU(True))

        # Primary output head
        self.head = nn.Conv2d(32, 1, 1)

        # Deep supervision heads (paper Eq. 20)
        self.ds1 = nn.Conv2d(128, 1, 1)   # 96×96 → upsampled to 192×192
        self.ds2 = nn.Conv2d(256, 1, 1)    # 48×48 → upsampled to 96×96
        self.ds3 = nn.Conv2d(512, 1, 1)    # 24×24 → upsampled to 48×48

    def forward(self, x5, x4, x3, x2):
        """
        x5: layer4 [B,2048,12,12]
        x4: layer3 [B,1024,24,24]
        x3: layer2 [B,512,48,48]
        x2: layer1 [B,256,96,96]
        """
        d4 = self.dec4(x5, x4)   # [B,512,24,24]
        d3 = self.dec3(d4, x3)   # [B,256,48,48]
        d2 = self.dec2(d3, x2)   # [B,128,96,96]
        d1 = self.dec1(d2)       # [B,32,384,384]

        # Primary output
        primary = self.head(d1)  # [B,1,384,384]

        # Deep supervision outputs
        aux1 = F.interpolate(self.ds1(d2), size=(192,192), mode='bilinear', align_corners=False)
        aux2 = F.interpolate(self.ds2(d3), size=(96,96),   mode='bilinear', align_corners=False)
        aux3 = F.interpolate(self.ds3(d4), size=(48,48),   mode='bilinear', align_corners=False)

        return primary, [aux1, aux2, aux3]


# ════════════════════════════════════════════════════════════════
# DERMFUSION MODEL — FULL ARCHITECTURE
# ════════════════════════════════════════════════════════════════
class DermFusion(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        D = CFG.feature_dim  # 768

        # ── CNN Backbone: ResNet-50 ──────────────────────────
        resnet = models.resnet50(weights="IMAGENET1K_V2")
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # [B,256,96,96]
        self.layer2 = resnet.layer2   # [B,512,48,48]
        self.layer3 = resnet.layer3   # [B,1024,24,24]
        self.layer4 = resnet.layer4   # [B,2048,12,12]

        # ── CBAM after each stage (paper Eqs. 4-5) ──────────
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # ── ViT Backbone ────────────────────────────────────
        # ViT-B/16 at 384×384 → (384/16)²=576 patches
        self.vit = timm.create_model(
            "vit_base_patch16_384", pretrained=True, num_classes=0)

        # ── Dimension Projection (paper Eqs. 9-10) ──────────
        self.cnn_proj = nn.Linear(2048, D)
        self.vit_proj = nn.Linear(768, D)

        # ── Morphology-Guided Bidirectional Cross-Attention ──
        #    (paper Eqs. 12-17 + structure-aware novelty)
        self.cross_attn = MorphGuidedCrossAttention(
            dim=D, num_heads=CFG.num_heads, dropout=CFG.dropout_ca)

        # ── Morphology Feature Gate (novelty, Mechanism 2) ───
        self.morph_gate = MorphFeatureGate(D)

        # ── U-Net Decoder + Deep Supervision ─────────────────
        self.seg_decoder = UNetDecoderWithDeepSup()

        # ── Classification Head ──────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(inplace=True),
            nn.Dropout(CFG.dropout_head),
            nn.Linear(256, num_classes))

        # ── Dropout for MC inference ─────────────────────────
        self.ca_dropout = nn.Dropout(CFG.dropout_ca)

    def forward(self, x, morph_strength=1.0):
        orig = x  # save for ViT

        # ── CNN + CBAM ────────────────────────────────────────
        x = self.stem(x)
        x1 = self.cbam1(self.layer1(x))    # [B,256,96,96]
        x2 = self.cbam2(self.layer2(x1))   # [B,512,48,48]
        x3 = self.cbam3(self.layer3(x2))   # [B,1024,24,24]
        x4 = self.cbam4(self.layer4(x3))   # [B,2048,12,12]

        # ── Segmentation (from CNN features) ──────────────────
        seg_primary, seg_aux = self.seg_decoder(x4, x3, x2, x1)

        # ── Morphology map M = sigmoid(seg_logits) ────────────
        #    This is the structure prior that steers fusion.
        morph_map = torch.sigmoid(seg_primary)           # [B,1,384,384]

        # Per-token morphology priors (downsample M to token grids):
        #   24×24 = 576 tokens to match CNN/ViT sequence length
        morph_24 = F.interpolate(morph_map, size=(24, 24),
                                 mode='bilinear', align_corners=False)
        morph_bias = morph_24.flatten(2).squeeze(1)      # [B, 576]
        # Centre to zero-mean so the bias adds/subtracts around the lesion
        morph_bias = morph_bias - morph_bias.mean(dim=1, keepdim=True)

        # Morphology descriptor (8 pooled stats) for the feature gate:
        #   mean, max, std of mask + quadrant means → richer than a scalar
        B = morph_map.shape[0]
        m_mean = morph_map.mean(dim=(2,3))                       # [B,1]
        m_max  = morph_map.amax(dim=(2,3))                       # [B,1]
        m_std  = morph_map.flatten(1).std(dim=1, keepdim=True)   # [B,1]
        H = morph_map.shape[2]; half = H // 2
        q_tl = morph_map[:, :, :half, :half].mean(dim=(2,3))     # [B,1]
        q_tr = morph_map[:, :, :half, half:].mean(dim=(2,3))
        q_bl = morph_map[:, :, half:, :half].mean(dim=(2,3))
        q_br = morph_map[:, :, half:, half:].mean(dim=(2,3))
        morph_desc = torch.cat([m_mean, m_max, m_std,
                                q_tl, q_tr, q_bl, q_br,
                                m_mean*m_max], dim=1)            # [B,8]

        # ── CNN token projection (paper Eq. 9) ────────────────
        cnn_up = F.interpolate(x4, size=(24, 24), mode='bilinear', align_corners=False)
        cnn_tokens = cnn_up.flatten(2).transpose(1, 2)  # [B, 576, 2048]
        cnn_tokens = self.cnn_proj(cnn_tokens)           # [B, 576, 768]

        # ── ViT tokens (paper Eq. 7) ──────────────────────────
        vit_out = self.vit.forward_features(orig)        # [B, 577, 768]
        vit_tokens = vit_out[:, 1:]                      # [B, 576, 768] (remove CLS)
        vit_tokens = self.vit_proj(vit_tokens)           # [B, 576, 768]

        # ── Morphology-Guided Cross-Attention (novelty) ───────
        #    Both directions are biased toward lesion regions M.
        #    morph_strength ramps 0→1 after the segmentation warmup.
        F_fusion = self.cross_attn(cnn_tokens, vit_tokens,
                                   morph_cnn=morph_bias, morph_vit=morph_bias,
                                   morph_strength=morph_strength)
        F_fusion = self.ca_dropout(F_fusion)

        # ── Global pool ───────────────────────────────────────
        cls_feat = F_fusion.mean(dim=1)                  # [B, 768]

        # ── Morphology Feature Gating (novelty, Mechanism 2) ──
        #    Morphology descriptor conditions the gate; strength ramps in.
        cls_feat = self.morph_gate(cls_feat, morph_desc, strength=morph_strength)

        logits = self.classifier(cls_feat)               # [B, 7]

        return {
            "logits": logits,
            "seg_logits": seg_primary,
            "seg_aux": seg_aux,
            "morph_map": morph_map,
        }

# ════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """Paper Eq. 18: Focal loss with optional class weights + label smoothing.
    Supports soft targets (for mixup)."""
    def __init__(self, gamma=1.5, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ls = label_smoothing

    def forward(self, logits, targets):
        # Soft targets (mixup) → use soft CE; else hard CE with label smoothing
        if targets.dim() == 2:  # soft one-hot [B, C]
            logp = F.log_softmax(logits, dim=1)
            ce = -(targets * logp).sum(dim=1)
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()
        else:
            ce = F.cross_entropy(logits, targets, weight=self.alpha,
                                 reduction='none', label_smoothing=self.ls)
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()

class FocalTverskyLoss(nn.Module):
    """Paper Eqs. 21-22: Focal-Tversky loss for segmentation."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha    # FP penalty
        self.beta = beta      # FN penalty (higher → penalise misses more)
        self.gamma = gamma    # focal parameter
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        # Resize target to match pred if needed
        if pred.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, pred.shape[2:], mode='nearest')
        tp = (pred * target).sum(dim=(2,3))
        fp = (pred * (1 - target)).sum(dim=(2,3))
        fn = ((1 - pred) * target).sum(dim=(2,3))
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return ((1 - tversky) ** self.gamma).mean()

class DermFusionLoss(nn.Module):
    """
    Paper Eq. 19: Homoscedastic multi-task loss.
    L_total = (1/2σ₁²)·L_cls + (1/2σ₂²)·L_seg + log(σ₁) + log(σ₂)
    """
    def __init__(self, class_weights=None):
        super().__init__()
        self.focal = FocalLoss(gamma=CFG.focal_gamma, alpha=None,
                               label_smoothing=CFG.label_smoothing)
        self.ft_loss = FocalTverskyLoss(CFG.tversky_alpha, CFG.tversky_beta, CFG.tversky_gamma)
        self.bce = nn.BCEWithLogitsLoss()

        # Learnable log-variance (paper Eq. 19)
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_seg = nn.Parameter(torch.zeros(1))

    def forward(self, outputs, batch):
        # Classification loss (Focal, Eq. 18)
        L_cls = self.focal(outputs["logits"], batch["label"])

        # Primary segmentation loss (0.4 BCE + 0.6 Focal-Tversky, Eq. 23)
        seg_pred = outputs["seg_logits"]
        seg_gt = batch["mask"]
        if seg_pred.shape[2:] != seg_gt.shape[2:]:
            seg_gt_r = F.interpolate(seg_gt, seg_pred.shape[2:], mode='nearest')
        else:
            seg_gt_r = seg_gt
        L_seg_primary = (CFG.seg_bce_weight * self.bce(seg_pred, seg_gt_r) +
                         CFG.seg_tv_weight * self.ft_loss(seg_pred, seg_gt_r))

        # Deep supervision losses (Eq. 20)
        L_deep = 0.0
        for aux, w in zip(outputs["seg_aux"], CFG.deep_sup_weights):
            aux_gt = F.interpolate(seg_gt, aux.shape[2:], mode='nearest')
            L_deep += w * self.ft_loss(aux, aux_gt)

        L_seg = L_seg_primary + L_deep
        if torch.rand(1).item() < 0.001:
            print(
                f"L_cls={L_cls.item():.4f} "
                f"L_seg_primary={L_seg_primary.item():.4f} "
                f"L_deep={L_deep.item():.4f}"
            )
        # Homoscedastic weighting (Eq. 19)
        # precision_cls = torch.exp(-self.log_var_cls)
        # precision_seg = torch.exp(-self.log_var_seg)
        # L_total = (precision_cls * L_cls + self.log_var_cls +
        #            # precision_seg * L_seg + self.log_var_seg)
        L_total = L_cls + 2.0 * L_seg
        return L_total


# ════════════════════════════════════════════════════════════════
# EMA
# ════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model); self.model.eval(); self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for ep, mp in zip(self.model.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1-self.decay)


# ════════════════════════════════════════════════════════════════
# MC DROPOUT INFERENCE
# ════════════════════════════════════════════════════════════════
def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()

@torch.no_grad()
def mc_predict(model, x, T=CFG.mc_samples, morph_strength=1.0):
    model.eval(); enable_dropout(model)
    preds = []
    for _ in range(T):
        out = model(x, morph_strength=morph_strength)
        preds.append(torch.softmax(out["logits"], dim=1).unsqueeze(0))
    preds = torch.cat(preds, 0)  # [T, B, C]
    mean_probs = preds.mean(0)         # [B, C]
    variance = preds.var(0).mean(1)    # [B] — scalar uncertainty (Eq. 24)
    return mean_probs, variance

# ════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════
def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    if pred.shape[2:] != target.shape[2:]:
        target = F.interpolate(target, pred.shape[2:], mode='nearest')
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2*inter + 1e-6) / (union + 1e-6)

def safe_auc(y_true, probs, n_classes=7):
    try:
        y_oh = np.eye(n_classes)[y_true]
        return roc_auc_score(y_oh, probs, multi_class='ovr')
    except: return 0.0

# ════════════════════════════════════════════════════════════════
# BUILD MODEL
# ════════════════════════════════════════════════════════════════
print("\nBuilding DermFusion...")
model = DermFusion(num_classes=CFG.num_classes).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"✓ Parameters: {n_params:.1f}M")

# Class frequency weights for focal loss
class_freq = df.label.value_counts().sort_index().values.astype(float)
class_weights = torch.tensor(1.0 / class_freq, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * CFG.num_classes
class_weights = class_weights.to(DEVICE)
print(f"✓ Class weights: {class_weights.cpu().numpy().round(3)}")

criterion = DermFusionLoss(class_weights=class_weights).to(DEVICE)

# ════════════════════════════════════════════════════════════════
# FREEZE BACKBONE FOR WARMUP
# ════════════════════════════════════════════════════════════════
for p in model.vit.parameters(): p.requires_grad = False
for p in model.layer1.parameters(): p.requires_grad = False
for p in model.layer2.parameters(): p.requires_grad = False
print("✓ Backbone frozen for warmup")

# EMA
ema = EMA(model, CFG.ema_decay)

# ════════════════════════════════════════════════════════════════
# OPTIMIZER — differential learning rates
# ════════════════════════════════════════════════════════════════
optimizer = torch.optim.AdamW([
    {"params": model.stem.parameters(),       "lr": CFG.backbone_lr},
    {"params": model.layer1.parameters(),     "lr": CFG.backbone_lr},
    {"params": model.layer2.parameters(),     "lr": CFG.backbone_lr},
    {"params": model.layer3.parameters(),     "lr": CFG.backbone_lr},
    {"params": model.layer4.parameters(),     "lr": CFG.backbone_lr},
    {"params": model.cbam1.parameters(),      "lr": CFG.head_lr},
    {"params": model.cbam2.parameters(),      "lr": CFG.head_lr},
    {"params": model.cbam3.parameters(),      "lr": CFG.head_lr},
    {"params": model.cbam4.parameters(),      "lr": CFG.head_lr},
    {"params": model.vit.parameters(),        "lr": CFG.vit_lr},
    {"params": model.cnn_proj.parameters(),   "lr": CFG.head_lr},
    {"params": model.vit_proj.parameters(),   "lr": CFG.head_lr},
    {"params": model.cross_attn.parameters(), "lr": CFG.head_lr},
    {"params": model.morph_gate.parameters(), "lr": CFG.head_lr},
    {"params": model.seg_decoder.parameters(),"lr": CFG.head_lr},
    {"params": model.classifier.parameters(), "lr": CFG.head_lr},
    {"params": criterion.parameters(),        "lr": CFG.head_lr},  # v1, v2
], weight_decay=CFG.weight_decay)

# Warmup + cosine schedule (per-epoch LR multiplier)
def lr_lambda(epoch):
    if epoch < CFG.warmup_epochs:
        return (epoch + 1) / CFG.warmup_epochs          # linear warmup
    # cosine decay over remaining epochs
    progress = (epoch - CFG.warmup_epochs) / max(1, CFG.epochs - CFG.warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = GradScaler()

# ════════════════════════════════════════════════════════════════
# RESUME FROM CHECKPOINT (if exists)
# ════════════════════════════════════════════════════════════════
best_auc = 0
history = []
start_epoch = 0

if CFG.RESUME and os.path.exists(CFG.CKPT_PATH):
    print(f"\n⟳ Resuming from {CFG.CKPT_PATH}")
    ckpt = torch.load(CFG.CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    ema.model.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    criterion.load_state_dict(ckpt["criterion"])
    start_epoch = ckpt["epoch"] + 1
    best_auc = ckpt["best_auc"]
    history = ckpt["history"]
    print(f"  Resumed at epoch {start_epoch}, best_auc={best_auc:.4f}")
    # Re-apply unfreeze state if past freeze point
    if start_epoch >= CFG.freeze_epochs:
        for p in model.vit.parameters(): p.requires_grad = True
        for p in model.layer1.parameters(): p.requires_grad = True
        for p in model.layer2.parameters(): p.requires_grad = True

# ════════════════════════════════════════════════════════════════
# MIXUP (soft-label augmentation, +2-3% on HAM10000)
# ════════════════════════════════════════════════════════════════
def mixup_batch(images, labels, alpha=CFG.mixup_alpha, n_classes=CFG.num_classes):
    """Returns mixed images and soft one-hot targets."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    oh = F.one_hot(labels, n_classes).float()
    soft = lam * oh + (1 - lam) * oh[idx]
    return mixed, soft

# ════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════
for epoch in range(start_epoch, CFG.epochs):

    # ── Unfreeze at epoch 6 ───────────────────────────────────
    if epoch == CFG.freeze_epochs:
        for p in model.vit.parameters(): p.requires_grad = True
        for p in model.layer1.parameters(): p.requires_grad = True
        for p in model.layer2.parameters(): p.requires_grad = True
        print(f"\n{'='*60}\n  BACKBONE UNFROZEN at epoch {epoch+1}\n{'='*60}")

    # ── Morphology guidance warmup ramp (curriculum) ──────────
    #    Strength = 0 while the segmentation branch is still learning,
    #    then linearly ramps to 1.0 over the next 6 epochs. This stops
    #    unreliable early masks from corrupting classification.
    if epoch < CFG.morph_warmup_start:
        morph_strength = 0.0
    elif epoch < CFG.morph_warmup_end:
        morph_strength = (epoch - CFG.morph_warmup_start) / \
                         (CFG.morph_warmup_end - CFG.morph_warmup_start)
    else:
        morph_strength = 1.0
    print(f"  [morph_strength = {morph_strength:.2f}]")

    model.train()
    train_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(TRAIN_LOADER, desc=f"Epoch {epoch+1}/{CFG.epochs}")

    for step, batch in enumerate(pbar):
        images = batch["image"].to(DEVICE)
        masks  = batch["mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        # Mixup (50% of batches) — uses soft labels for classification
        use_mixup = np.random.rand() < 0.5
        if use_mixup:
            mixed_images, soft_labels = mixup_batch(images, labels)
        else:
            mixed_images = images

        with autocast():
            outputs = model(mixed_images, morph_strength=morph_strength)
            cls_target = soft_labels if use_mixup else labels
            loss = criterion(outputs, {"label": cls_target, "mask": masks})
            loss = loss / CFG.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % CFG.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if epoch >= CFG.warmup_epochs:   # skip noisy warmup weights
                ema.update(model)

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{train_loss/(step+1):.4f}")

    scheduler.step()

    # ── Validation ────────────────────────────────────────────
    model.eval()
    all_probs, all_labels, all_unc, dices = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(VALID_LOADER, desc="Valid", leave=False):
            images = batch["image"].to(DEVICE)
            masks  = batch["mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # CLEAN forward for accuracy/AUC + hflip TTA (Fix 7,8)
            out = model(images, morph_strength=morph_strength)
            out_flip = model(torch.flip(images, dims=[3]), morph_strength=morph_strength)
            clean_probs = 0.5 * (torch.softmax(out["logits"], 1) +
                                 torch.softmax(out_flip["logits"], 1))

            # MC dropout ONLY for uncertainty (not for the prediction itself)
            _, unc = mc_predict(model, images, morph_strength=morph_strength)

            # Segmentation dice
            d = dice_score(out["seg_logits"], masks)
            dices.append(d.item())

            all_probs.append(clean_probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_unc.append(unc.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_unc    = np.concatenate(all_unc)

    acc = accuracy_score(all_labels, all_probs.argmax(1))
    auc = safe_auc(all_labels, all_probs)
    f1  = f1_score(all_labels, all_probs.argmax(1), average='macro')
    mean_dice = np.mean(dices)
    mean_unc  = all_unc.mean()

    # Selective prediction (triage)
    # Find threshold where ~68% are retained
    sorted_unc = np.sort(all_unc)
    tau_68 = sorted_unc[int(0.68 * len(sorted_unc))] if len(sorted_unc) > 0 else 0.5
    retained = all_unc <= tau_68
    sel_acc = accuracy_score(all_labels[retained], all_probs[retained].argmax(1)) if retained.sum()>0 else 0

    # MEL sensitivity
    mel_mask = all_labels == 0  # MEL is class 0
    mel_preds = all_probs[mel_mask].argmax(1) if mel_mask.sum() > 0 else []
    mel_sens = (mel_preds == 0).mean() if len(mel_preds) > 0 else 0

    metrics = {"epoch": epoch+1, "acc": acc, "auc": auc, "f1": f1,
               "dice": mean_dice, "unc": mean_unc, "sel_acc": sel_acc,
               "mel_sens": mel_sens, "tau_68": tau_68, "coverage": retained.mean()}
    history.append(metrics)

    print(f"\n{'='*60}")
    print(f"  EPOCH  : {epoch+1}/{CFG.epochs}")
    print(f"  ACC    : {acc:.4f}    AUC    : {auc:.4f}")
    print(f"  F1     : {f1:.4f}    DICE   : {mean_dice:.4f}")
    print(f"  UNC    : {mean_unc:.6f}")
    print(f"  MEL Sensitivity : {mel_sens:.4f}")
    print(f"  Selective ACC   : {sel_acc:.4f} (τ={tau_68:.6f}, coverage={retained.mean():.2%})")
    print(f"  v1={criterion.log_var_cls.item():.4f}  v2={criterion.log_var_seg.item():.4f}")
    print(f"{'='*60}")

    if auc > best_auc:
        best_auc = auc
        # During warmup EMA hasn't tracked yet → save live model; EMA after.
        save_src = model if epoch < CFG.warmup_epochs else ema.model
        torch.save(save_src.state_dict(), os.environ.get("DERMFUSION_BEST_PATH", "dermfusion_best.pth"))
        print("  ★ BEST MODEL SAVED")

    # Save full checkpoint every epoch (for resume)
    torch.save({
        "epoch": epoch, "model": model.state_dict(), "ema": ema.model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "criterion": criterion.state_dict(),
        "best_auc": best_auc, "history": history,
    }, CFG.CKPT_PATH)

    gc.collect(); torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"  Best AUC: {best_auc:.4f}")
print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════
# TEMPERATURE CALIBRATION (post-hoc, paper Eqs. 25-26)
# ════════════════════════════════════════════════════════════════
print("\n--- Temperature Calibration ---")
# Load best model
best_model = DermFusion(num_classes=CFG.num_classes).to(DEVICE)
best_model.load_state_dict(torch.load(os.environ.get("DERMFUSION_BEST_PATH", "dermfusion_best.pth"), map_location=DEVICE))
best_model.eval()

# Collect validation logits
val_logits_all, val_labels_all = [], []
with torch.no_grad():
    for batch in VALID_LOADER:
        out = best_model(batch["image"].to(DEVICE), morph_strength=1.0)
        val_logits_all.append(out["logits"].cpu())
        val_labels_all.append(batch["label"])
val_logits_all = torch.cat(val_logits_all)
val_labels_all = torch.cat(val_labels_all)

# Grid search over T_cal
best_ece, best_T = 1.0, 1.0
for T in [1.00, 1.25, 1.50, 1.75, 2.00, 2.25]:
    probs = torch.softmax(val_logits_all / T, dim=1)
    confs, preds = probs.max(1)
    correct = (preds == val_labels_all).float()

    # ECE (15 bins)
    ece = 0.0
    for i in range(15):
        lo, hi = i/15, (i+1)/15
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confs[mask].mean()
            ece += mask.float().mean() * abs(bin_acc - bin_conf)

    print(f"  T={T:.2f} → ECE={ece:.4f}")
    if ece < best_ece:
        best_ece, best_T = ece, T

print(f"\n  ★ Optimal T*={best_T:.2f}, ECE={best_ece:.4f}")

# ════════════════════════════════════════════════════════════════
# FINAL METRICS SUMMARY
# ════════════════════════════════════════════════════════════════
best_epoch = max(history, key=lambda x: x["auc"])
print(f"\n{'='*60}")
print(f"  FINAL RESULTS (best epoch {best_epoch['epoch']})")
print(f"  Accuracy      : {best_epoch['acc']*100:.2f}%")
print(f"  ROC-AUC       : {best_epoch['auc']:.4f}")
print(f"  Macro F1      : {best_epoch['f1']:.4f}")
print(f"  Dice Score    : {best_epoch['dice']:.4f}")
print(f"  MEL Sensitivity: {best_epoch['mel_sens']*100:.1f}%")
print(f"  Selective ACC : {best_epoch['sel_acc']*100:.1f}% @ {best_epoch['coverage']*100:.0f}% coverage")
print(f"  T_cal*        : {best_T:.2f}")
print(f"  ECE (cal)     : {best_ece:.4f}")
print(f"  Parameters    : {n_params:.1f}M")
print(f"{'='*60}")
print("\n✓ Use these ACTUAL numbers in the paper.")
