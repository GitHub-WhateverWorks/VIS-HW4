# HW4 — Blind Image Restoration (Rain & Snow Removal)

Model: **PromptIR** with Restormer backbone, trained from scratch.  
Task: All-in-one blind restoration — a single model handles both rain and snow degradation without knowing the degradation type at test time.

---

## Environment

```bash
pip install torch torchvision matplotlib numpy Pillow
```

Python 3.10+, PyTorch 2.x. GPU required (A100 recommended for training; T4 sufficient for inference).

---

## Dataset

Place `hw4_realse_dataset.zip` inside `vis_hw4/`:

```
vis_hw4/
├── hw4_realse_dataset.zip
└── train.py
```

The script extracts it automatically on first run.

---

## Training

Two stages are run end-to-end with a single command:

```bash
python vis_hw4/train.py \
    --data_root  vis_hw4 \
    --ckpt_dir   vis_hw4/checkpoints \
    --pred_dir   vis_hw4/predictions \
    --plot_dir   vis_hw4/plots \
    --dim        64 \
    --prompt_len 8 \
    --crop       192 \
    --batch      2 \
    --s1_epochs  180 \
    --s2_epochs  40
```

| | Stage 1 | Stage 2 |
|---|---|---|
| Purpose | Base training | Fine-tune |
| Epochs | 180 | 40 |
| LR | 2e-4 | 5e-5 |
| LR schedule | Linear warmup (15 ep) + cosine | Linear warmup (3 ep) + cosine |
| EMA decay | 0.999 | 0.9995 |
| Augmentation | hflip + vflip + rot90 | hflip only |
| Crop size | 192×192 | 192×192 |
| Batch size | 2 | 2 |

Other fixed settings: AdamW (β=0.9, 0.999, wd=1e-5), Charbonnier loss (ε=1e-3), AMP (fp16), grad clip norm=1.0, train/val split=95/5.

Training auto-resumes from the last checkpoint if interrupted — just re-run the same command.

---

## Inference Only

If weights are already trained:

```bash
python vis_hw4/train.py \
    --data_root vis_hw4 \
    --ckpt_dir  vis_hw4/checkpoints \
    --pred_dir  vis_hw4/predictions \
    --dim       64 \
    --prompt_len 8 \
    --infer_only
```

Output is saved to `vis_hw4/predictions/pred.npz`.  
Each entry is keyed by filename (e.g. `"1.png"`) with shape `(3, H, W)`, dtype `uint8`.

```python
import numpy as np
data = np.load('vis_hw4/predictions/pred.npz')
arr = data['1.png']   # shape (3, H, W), uint8
```

Inference uses 4-fold TTA (original + hflip + vflip + hvflip).

---

## Regenerate Plots

```bash
python vis_hw4/train.py \
    --ckpt_dir vis_hw4/checkpoints \
    --plot_dir vis_hw4/plots \
    --plots_only
```

Saves 6 figures to `vis_hw4/plots/`:

| File | Content |
|---|---|
| `psnr_stage1.png` | Stage 1 validation PSNR — Rain / Snow / Total |
| `loss_stage1.png` | Stage 1 Charbonnier loss |
| `psnr_stage2.png` | Stage 2 validation PSNR — Rain / Snow / Total |
| `loss_stage2.png` | Stage 2 Charbonnier loss |
| `psnr_final_combined.png` | Full training PSNR across all stages |
| `loss_final_combined.png` | Full training loss across all stages |

---

## Google Colab

Open `train_colab.ipynb` in Colab. Set runtime to **A100** (Pro/Pro+) for training, **T4** is sufficient for inference only.

The notebook has two modes:
- **Full training** — runs both stages end-to-end, saves checkpoints to Drive after each best epoch
- **Recovery** — loads existing weights + `history.json` from Drive, regenerates plots and `pred.npz` without re-training

---

## Model Architecture

**Backbone:** Restormer — 4-level UNet with Transformer blocks (MDTA + GDFN)  
**Prompt blocks:** 3 PromptIR blocks in the decoder, each with a Prompt Generation Module (learned degradation dictionary, `prompt_len=8`) and Prompt Interaction Module  
**Parameters:** ~48.6M (dim=64)  
**Key design:** No degradation label is used at test time — the PGM implicitly identifies the degradation type via a weighted combination of learned prompt components

---

## Output Structure

```
vis_hw4/
├── checkpoints/
│   ├── stage1_best.pth      # best EMA weights from stage 1
│   ├── stage2_best.pth      # best EMA weights from stage 2
│   └── history.json         # per-epoch loss/PSNR for all stages
├── predictions/
│   └── pred.npz             # test set outputs
├── plots/
│   ├── psnr_stage1.png
│   ├── loss_stage1.png
│   ├── psnr_stage2.png
│   ├── loss_stage2.png
│   ├── psnr_final_combined.png
│   └── loss_final_combined.png
├── train.py
├── train_colab.ipynb
└── README.md
```
