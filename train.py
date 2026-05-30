"""
train.py  —  PromptIR (Restormer-64) for blind rain+snow removal
Two stages:
  Stage 1 : 180 epochs, full aug (hflip+vflip+rot90), lr=2e-4, EMA=0.999
  Stage 2 : 40 epochs fine-tune, hflip only, lr=5e-5, EMA=0.9995
Dataset expected at: vis_hw4/hw4_realse_dataset.zip  (auto-extracted)
"""

import argparse
import json
import math
import random
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

# ─────────────────────────────── helpers ───────────────────────────────────


def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def basename_map(paths: List[Path]):
    return {p.name.lower(): p for p in paths}


def find_pairs(
    data_root: Path,
) -> Tuple[List[Tuple[Path, Path, str]], List[Path]]:
    paths = [
        p for p in list_images(data_root) if "__macosx" not in str(p).lower()
    ]
    by_name = basename_map(paths)

    pairs: List[Tuple[Path, Path, str]] = []
    for typ in ("rain", "snow"):
        for idx in range(1, 1601):
            deg_name = f"{typ}-{idx}.png"
            clean_name = f"{typ}_clean-{idx}.png"
            deg = by_name.get(deg_name)
            clean = by_name.get(clean_name)
            if deg is not None and clean is not None:
                pairs.append((deg, clean, typ))

    if not pairs:
        raise RuntimeError(
            "No train pairs found. Expected names like "
            "rain-1.png / rain_clean-1.png and "
            "snow-1.png / snow_clean-1.png somewhere under "
            "the extracted dataset."
        )

    numeric_test: List[Path] = []
    train_names = {p.name.lower() for pair in pairs for p in pair[:2]}
    for p in paths:
        if p.name.lower() in train_names:
            continue
        if p.stem.isdigit():
            numeric_test.append(p)
    numeric_test = sorted(numeric_test, key=lambda x: int(x.stem))
    return pairs, numeric_test


def extract_zip(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path} → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print("Extraction done.")


# ─────────────────────────────── dataset ───────────────────────────────────


class RestorationDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path, str]],
        crop: int = 192,
        augment: str = "full",
    ):  # "full" | "hflip"
        self.pairs = pairs
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, clean_path, typ = self.pairs[idx]
        deg = Image.open(deg_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        # random crop (same region for both)
        i, j, h, w = transforms.RandomCrop.get_params(
            deg, (self.crop, self.crop)
        )
        deg = TF.crop(deg, i, j, h, w)
        clean = TF.crop(clean, i, j, h, w)

        # augmentation
        if random.random() < 0.5:
            deg, clean = TF.hflip(deg), TF.hflip(clean)
        if self.augment == "full":
            if random.random() < 0.5:
                deg, clean = TF.vflip(deg), TF.vflip(clean)
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                deg = TF.rotate(deg, 90 * k)
                clean = TF.rotate(clean, 90 * k)

        deg = TF.to_tensor(deg)
        clean = TF.to_tensor(clean)
        return deg, clean, typ


class ValDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, clean_path, typ = self.pairs[idx]
        deg = TF.to_tensor(Image.open(deg_path).convert("RGB"))
        clean = TF.to_tensor(Image.open(clean_path).convert("RGB"))
        return deg, clean, typ


# ──────────────────────── model architecture ───────────────────────────────
#  Restormer + PromptIR blocks (self-contained, no external dependency)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention"""

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_d = nn.Conv2d(
            dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias
        )
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_d(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, -1, H * W)
        k = k.reshape(B, self.num_heads, -1, H * W)
        v = v.reshape(B, self.num_heads, -1, H * W)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, -1, H, W)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""

    def __init__(self, dim, ffn_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_factor)
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias
        )
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.proj_in(x)
        x1, x2 = self.dw(x).chunk(2, dim=1)
        return self.proj_out(x1 * F.gelu(x2))


class TransformerBlock(nn.Module):
    def __init__(
        self, dim, num_heads, ffn_factor=2.66, bias=False, ln_bias=True
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=ln_bias)
        self.ffn = GDFN(dim, ffn_factor, bias)

    def forward(self, x):
        B, C, H, W = x.shape
        # layer norm over channel dim
        x = x + self.attn(
            self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        )
        x = x + self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


def make_encoder_level(dim, n_blocks, num_heads):
    return nn.Sequential(
        *[TransformerBlock(dim, num_heads) for _ in range(n_blocks)]
    )


class Downsample(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * 2, 3, padding=1, bias=False)
        self.ps = nn.PixelUnshuffle(
            2
        )  # (B, in_c*2, H, W) → (B, in_c*8, H/2, W/2)

    def forward(self, x):
        return self.ps(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * 2, 3, padding=1, bias=False)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.conv(x))


# ── Prompt modules ──────────────────────────────────────────────────────────


class PromptGenModule(nn.Module):
    def __init__(self, dim, prompt_len=8):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompts = nn.Parameter(torch.randn(prompt_len, dim, 1, 1))
        self.fc = nn.Linear(dim, prompt_len)

    def forward(self, x):
        # x: (B, C, H, W)
        w = self.fc(x.mean(dim=[2, 3]))  # (B, prompt_len)
        w = w.softmax(dim=-1)  # (B, prompt_len)
        # self.prompts: (prompt_len, C, 1, 1)
        # unsqueeze(0) -> (1, prompt_len, C, 1, 1)
        # w[:,:,None,None,None] -> (B, prompt_len, 1, 1, 1)
        # product -> (B, prompt_len, C, 1, 1), sum(dim=1) -> (B, C, 1, 1)
        p = (w[:, :, None, None, None] * self.prompts.unsqueeze(0)).sum(dim=1)
        p = p.expand(-1, -1, x.shape[2], x.shape[3])
        return p


class PromptBlock(nn.Module):
    def __init__(self, dim, num_heads, prompt_len=8):
        super().__init__()
        self.pgm = PromptGenModule(dim, prompt_len)
        self.fuse = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.block = TransformerBlock(dim, num_heads)

    def forward(self, x):
        p = self.pgm(x)
        out = self.block(self.fuse(torch.cat([x, p], dim=1)))
        return out


# ── Full PromptIR-Restormer ──────────────────────────────────────────────────


class PromptIR(nn.Module):
    def __init__(
        self,
        dim=64,
        num_blocks=(4, 6, 6, 8),
        num_heads=(1, 2, 4, 8),
        ffn_factor=2.66,
        prompt_len=8,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 3, padding=1, bias=False)

        # Encoder
        self.enc1 = make_encoder_level(dim, num_blocks[0], num_heads[0])
        self.enc2 = make_encoder_level(dim * 2, num_blocks[1], num_heads[1])
        self.enc3 = make_encoder_level(dim * 4, num_blocks[2], num_heads[2])

        self.down1 = Downsample(
            dim
        )  # dim   → dim*2  (but PixelUnshuffle makes dim*8 → need fix)
        self.down2 = Downsample(dim * 2)
        self.down3 = Downsample(dim * 4)
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1, bias=False)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 4, stride=2, padding=1, bias=False)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, 4, stride=2, padding=1, bias=False)
        )

        # Bottleneck
        self.bottleneck = make_encoder_level(
            dim * 8, num_blocks[3], num_heads[3]
        )

        # Decoder (with prompt blocks at each level)
        self.up3 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 4 * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.skip3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.dec3 = make_encoder_level(dim * 4, num_blocks[2], num_heads[2])
        self.prompt3 = PromptBlock(
            dim * 4, num_heads[2], prompt_len
        )  # channels = dim*4

        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2 * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.skip2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.dec2 = make_encoder_level(dim * 2, num_blocks[1], num_heads[1])
        self.prompt2 = PromptBlock(
            dim * 2, num_heads[1], prompt_len
        )  # channels = dim*2

        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.skip1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = make_encoder_level(dim, num_blocks[0], num_heads[0])
        self.prompt1 = PromptBlock(
            dim, num_heads[0], prompt_len
        )  # channels = dim

        # Refinement
        self.refine = make_encoder_level(dim, 4, num_heads[0])

        # Output
        self.output = nn.Conv2d(dim, 3, 3, padding=1, bias=False)

    def forward(self, x):
        inp = x
        x = self.patch_embed(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.prompt3(
            self.dec3(self.skip3(torch.cat([self.up3(b), e3], dim=1)))
        )
        d2 = self.prompt2(
            self.dec2(self.skip2(torch.cat([self.up2(d3), e2], dim=1)))
        )
        d1 = self.prompt1(
            self.dec1(self.skip1(torch.cat([self.up1(d2), e1], dim=1)))
        )

        out = self.output(self.refine(d1)) + inp
        return out


# ─────────────────────────────── loss ──────────────────────────────────────


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps**2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


# ─────────────────────────────── EMA ───────────────────────────────────────


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def set_decay(self, decay: float):
        self.decay = decay


# ─────────────────────────────── PSNR ──────────────────────────────────────


def psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse.item())


# ─────────────────────────────── TTA ───────────────────────────────────────


@torch.no_grad()
def tta_predict(model, x):
    preds = []
    for hf in [False, True]:
        for vf in [False, True]:
            xi = x
            if hf:
                xi = xi.flip(-1)
            if vf:
                xi = xi.flip(-2)
            p = model(xi)
            if vf:
                p = p.flip(-2)
            if hf:
                p = p.flip(-1)
            preds.append(p)
    return torch.stack(preds).mean(0)


# ─────────────────────────────── eval ──────────────────────────────────────


@torch.no_grad()
def evaluate(model, loader, device, use_tta=False):
    model.eval()
    rain_psnr, snow_psnr = [], []
    for deg, clean, typs in loader:
        deg, clean = deg.to(device), clean.to(device)
        if use_tta:
            pred = tta_predict(model, deg)
        else:
            pred = model(deg)
        pred = pred.clamp(0, 1)
        for i, t in enumerate(typs):
            v = psnr(pred[i: i + 1], clean[i: i + 1])
            if t == "rain":
                rain_psnr.append(v)
            else:
                snow_psnr.append(v)
    r = float(np.mean(rain_psnr)) if rain_psnr else 0.0
    s = float(np.mean(snow_psnr)) if snow_psnr else 0.0
    tot = float(np.mean(rain_psnr + snow_psnr))
    return r, s, tot


# ─────────────────────────── LR schedule ───────────────────────────────────


def build_scheduler(optimizer, warmup_epochs, total_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(
            total_epochs - warmup_epochs, 1
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_epoch
    )


# ─────────────────────────────── train loop ────────────────────────────────


def run_stage(
    model,
    ema,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    warmup,
    ema_decay,
    ckpt_dir: Path,
    stage_name: str,
    history: dict,
    resume_ckpt: Path = None,
):
    ema.set_decay(ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5
    )
    scaler = GradScaler("cuda")
    criterion = CharbonnierLoss(eps=1e-3)
    scheduler = build_scheduler(optimizer, warmup, epochs)

    start_epoch = 0
    best_psnr = 0.0
    best_ckpt = ckpt_dir / f"{stage_name}_best.pth"

    if resume_ckpt and resume_ckpt.exists():
        print(f"Resuming {stage_name} from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        history.update(ckpt.get("history", {}))
        print(f"  → resumed at epoch {start_epoch}, best={best_psnr:.4f}")

    stage_losses, stage_rain, stage_snow, stage_tot = [], [], [], []

    for epoch in range(start_epoch, epochs):
        model.train()
        ep_losses = []
        t0 = time.time()
        for deg, clean, _ in train_loader:
            deg, clean = deg.to(device), clean.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(deg)
                loss = criterion(pred, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            ep_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(ep_losses))

        # validation (EMA model)
        r, s, tot = evaluate(ema.model, val_loader, device)
        stage_losses.append(avg_loss)
        stage_rain.append(r)
        stage_snow.append(s)
        stage_tot.append(tot)

        elapsed = time.time() - t0
        print(
            f"[{stage_name}] Ep {epoch + 1:4d}/{epochs}  "
            f"loss={avg_loss:.5f}  "
            f"PSNR rain={r:.2f} snow={s:.2f} tot={tot:.2f}  "
            f"({elapsed:.0f}s)"
        )

        if tot > best_psnr:
            best_psnr = tot
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "history": history,
                },
                best_ckpt,
            )
            print(f"  ✓ saved best ({best_psnr:.4f} dB) → {best_ckpt}")

        # periodic save
        if (epoch + 1) % 20 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "history": history,
                },
                ckpt_dir / f"{stage_name}_ep{epoch + 1}.pth",
            )

    history[stage_name] = {
        "loss": stage_losses,
        "rain": stage_rain,
        "snow": stage_snow,
        "total": stage_tot,
    }
    json.dump(history, open(ckpt_dir / "history.json", "w"), indent=2)
    return best_ckpt, history


# ─────────────────────────────── inference ─────────────────────────────────


@torch.no_grad()
def run_inference(model, test_paths: List[Path], out_dir: Path, device):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dict = {}
    for p in test_paths:
        img = (
            TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        )
        pred = tta_predict(model, img).squeeze(0).clamp(0, 1)
        # (3, H, W) uint8 — matches the expected npz format
        arr = (pred.cpu().numpy() * 255).round().astype(np.uint8)
        images_dict[p.name] = arr
    npz_path = out_dir / "pred.npz"
    np.savez(npz_path, **images_dict)
    print(f"Saved {len(images_dict)} images to {npz_path}")


# ─────────────────────────────── plots ─────────────────────────────────────


def _plot_one_stage_psnr(ax, data, title, colour):
    """Plot rain/snow/total PSNR for a single stage onto ax."""
    eps = range(1, len(data["total"]) + 1)
    ax.plot(eps, data["total"], color=colour, label="Total", linewidth=2)
    ax.plot(
        eps,
        data["rain"],
        color="#E91E63",
        label="Rain",
        linewidth=1.5,
        linestyle="--",
    )
    ax.plot(
        eps,
        data["snow"],
        color="#00BCD4",
        label="Snow",
        linewidth=1.5,
        linestyle=":",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    lo = max(
        0, min(min(data["rain"]), min(data["snow"]), min(data["total"])) - 0.5
    )
    hi = max(max(data["rain"]), max(data["snow"]), max(data["total"])) + 0.3
    ax.set_ylim(lo, hi)


def _plot_one_stage_loss(ax, data, title, colour):
    """Plot Charbonnier loss for a single stage onto ax."""
    eps = range(1, len(data["loss"]) + 1)
    ax.plot(eps, data["loss"], color=colour, linewidth=1.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Charbonnier Loss")
    ax.grid(alpha=0.3)


def plot_history(history: dict, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found — skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    COLOURS = {"stage1": "#2196F3", "stage2": "#FF5722"}
    LABELS = {
        "stage1": "Stage 1 — Base Training (180 ep)",
        "stage2": "Stage 2 — Fine-tune (40 ep)",
    }

    # ── Figure 1: Stage 1 PSNR ───────────────────────────────────────────────
    if "stage1" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_one_stage_psnr(
            ax, history["stage1"], LABELS["stage1"], COLOURS["stage1"]
        )
        plt.tight_layout()
        p = out_dir / "psnr_stage1.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {p}")

    # ── Figure 2: Stage 1 Loss ───────────────────────────────────────────────
    if "stage1" in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_one_stage_loss(
            ax, history["stage1"], LABELS["stage1"], COLOURS["stage1"]
        )
        plt.tight_layout()
        p = out_dir / "loss_stage1.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {p}")

    # ── Figure 3: Stage 2 PSNR ───────────────────────────────────────────────
    if "stage2" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_one_stage_psnr(
            ax, history["stage2"], LABELS["stage2"], COLOURS["stage2"]
        )
        plt.tight_layout()
        p = out_dir / "psnr_stage2.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {p}")

    # ── Figure 4: Stage 2 Loss ───────────────────────────────────────────────
    if "stage2" in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        _plot_one_stage_loss(
            ax, history["stage2"], LABELS["stage2"], COLOURS["stage2"]
        )
        plt.tight_layout()
        p = out_dir / "loss_stage2.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {p}")

    # ── Figure 5: Combined PSNR across all stages ────────────────────────────
    if len(history) >= 1:
        all_rain, all_snow, all_tot = [], [], []
        boundaries = [0]
        stage_names = list(history.keys())
        for sname in stage_names:
            data = history[sname]
            all_rain.extend(data["rain"])
            all_snow.extend(data["snow"])
            all_tot.extend(data["total"])
            boundaries.append(len(all_tot))

        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(1, len(all_tot) + 1)
        ax.plot(x, all_tot, color="#333333", label="Total", linewidth=2)
        ax.plot(
            x,
            all_rain,
            color="#E91E63",
            label="Rain",
            linewidth=1.5,
            linestyle="--",
        )
        ax.plot(
            x,
            all_snow,
            color="#00BCD4",
            label="Snow",
            linewidth=1.5,
            linestyle=":",
        )

        # shade background per stage
        stage_colours_bg = {"stage1": "#E3F2FD", "stage2": "#FBE9E7"}
        for i, sname in enumerate(stage_names):
            x0, x1 = boundaries[i] + 1, boundaries[i + 1]
            bg = stage_colours_bg.get(sname, "#F5F5F5")
            ax.axvspan(
                x0, x1, alpha=0.25, color=bg, label=LABELS.get(sname, sname)
            )
            # stage label at top
            mid = (x0 + x1) / 2
            ax.text(
                mid,
                ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 32,
                LABELS.get(sname, sname).split("—")[0].strip(),
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLOURS.get(sname, "#555"),
            )

        ax.set_xlabel("Epoch (cumulative)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(
            "Full Training — PSNR Across All Stages (Rain / Snow / Total)",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p = out_dir / "psnr_final_combined.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {p}")

    # ── Figure 6: Combined Loss across all stages ────────────────────────────
    if len(history) >= 1:
        all_loss = []
        boundaries = [0]
        for sname in stage_names:
            all_loss.extend(history[sname]["loss"])
            boundaries.append(len(all_loss))

        fig, ax = plt.subplots(figsize=(12, 4))
        x = range(1, len(all_loss) + 1)
        ax.plot(x, all_loss, color="#37474F", linewidth=1.6)

        for i, sname in enumerate(stage_names):
            x0, x1 = boundaries[i] + 1, boundaries[i + 1]
            bg = stage_colours_bg.get(sname, "#F5F5F5")
            ax.axvspan(x0, x1, alpha=0.25, color=bg)
            mid = (x0 + x1) / 2
            ax.text(
                mid,
                max(all_loss) * 0.98,
                LABELS.get(sname, sname).split("—")[0].strip(),
                ha="center",
                va="top",
                fontsize=9,
                color=COLOURS.get(sname, "#555"),
            )

        ax.set_xlabel("Epoch (cumulative)")
        ax.set_ylabel("Charbonnier Loss")
        ax.set_title(
            "Full Training — Loss Across All Stages",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p = out_dir / "loss_final_combined.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {p}")


# ─────────────────────────────── main ─────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        default="vis_hw4",
        help="Folder containing hw4_realse_dataset.zip (or already extracted)",
    )
    p.add_argument("--ckpt_dir", default="vis_hw4/checkpoints")
    p.add_argument("--pred_dir", default="vis_hw4/predictions")
    p.add_argument("--plot_dir", default="vis_hw4/plots")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--prompt_len", type=int, default=8)
    p.add_argument("--crop", type=int, default=192)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    # Stage 1
    p.add_argument("--s1_epochs", type=int, default=180)
    p.add_argument("--s1_lr", type=float, default=2e-4)
    p.add_argument("--s1_warmup", type=int, default=15)
    p.add_argument("--s1_ema", type=float, default=0.999)
    # Stage 2
    p.add_argument("--s2_epochs", type=int, default=40)
    p.add_argument("--s2_lr", type=float, default=5e-5)
    p.add_argument("--s2_warmup", type=int, default=3)
    p.add_argument("--s2_ema", type=float, default=0.9995)
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--infer_only",
        action="store_true",
        help="Skip training; only run inference with best checkpoint",
    )
    p.add_argument(
        "--plots_only",
        action="store_true",
        help="Re-generate plots from saved history.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.pred_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    # ── Plots-only mode ──────────────────────────────────────────────────────
    if args.plots_only:
        hist_path = ckpt_dir / "history.json"
        if not hist_path.exists():
            raise FileNotFoundError(f"{hist_path} not found.")
        history = json.load(open(hist_path))
        plot_history(history, plot_dir)
        return

    # ── Extract dataset ──────────────────────────────────────────────────────
    zip_path = data_root / "hw4_realse_dataset.zip"
    ext_dir = data_root / "dataset"
    if not ext_dir.exists():
        if zip_path.exists():
            extract_zip(zip_path, ext_dir)
        else:
            raise FileNotFoundError(
                f"Dataset zip not found at {zip_path}. "
                "Please place hw4_realse_dataset.zip inside vis_hw4/."
            )

    pairs, test_paths = find_pairs(ext_dir)
    print(f"Found {len(pairs)} train pairs  |  {len(test_paths)} test images")

    # 95/5 split
    random.shuffle(pairs)
    n_train = int(len(pairs) * 0.95)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    val_ds = ValDataset(val_pairs)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = PromptIR(dim=args.dim, prompt_len=args.prompt_len).to(device)
    ema = EMA(model, decay=args.s1_ema)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {total_params:.1f}M")

    history: dict = {}
    hist_path = ckpt_dir / "history.json"
    if hist_path.exists():
        history = json.load(open(hist_path))

    if not args.infer_only:
        # ── Stage 1 ──────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"STAGE 1 — Base Training ({args.s1_epochs} epochs)")
        print("=" * 60)
        train_ds1 = RestorationDataset(
            train_pairs, crop=args.crop, augment="full"
        )
        train_loader1 = DataLoader(
            train_ds1,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        s1_best, history = run_stage(
            model,
            ema,
            train_loader1,
            val_loader,
            device,
            epochs=args.s1_epochs,
            lr=args.s1_lr,
            warmup=args.s1_warmup,
            ema_decay=args.s1_ema,
            ckpt_dir=ckpt_dir,
            stage_name="stage1",
            history=history,
        )

        # Load best weights for stage 2
        print("\nLoading stage-1 best EMA → stage 2 init")
        ckpt = torch.load(s1_best, map_location=device)
        model.load_state_dict(ckpt["ema"])
        ema = EMA(model, decay=args.s2_ema)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"STAGE 2 — Fine-tuning ({args.s2_epochs} epochs, hflip only)")
        print("=" * 60)
        train_ds2 = RestorationDataset(
            train_pairs, crop=args.crop, augment="hflip"
        )
        train_loader2 = DataLoader(
            train_ds2,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        s2_best, history = run_stage(
            model,
            ema,
            train_loader2,
            val_loader,
            device,
            epochs=args.s2_epochs,
            lr=args.s2_lr,
            warmup=args.s2_warmup,
            ema_decay=args.s2_ema,
            ckpt_dir=ckpt_dir,
            stage_name="stage2",
            history=history,
        )
        final_ckpt = s2_best
    else:
        # find best available checkpoint
        s2_best = ckpt_dir / "stage2_best.pth"
        s1_best = ckpt_dir / "stage1_best.pth"
        final_ckpt = s2_best if s2_best.exists() else s1_best
        print(f"Inference-only mode, loading {final_ckpt}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if history:
        plot_history(history, plot_dir)

    # ── Inference on test set ────────────────────────────────────────────────
    if test_paths:
        print(f"\nRunning TTA inference on {len(test_paths)} test images …")
        ckpt = torch.load(final_ckpt, map_location=device)
        model.load_state_dict(ckpt["ema"])
        run_inference(model, test_paths, pred_dir, device)
    else:
        print("No test images found — skipping inference.")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
