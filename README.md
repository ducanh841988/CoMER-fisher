<div align="center">    
 
# CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition  
 
[![arXiv](https://img.shields.io/badge/arXiv-2207.04410-b31b1b.svg)](https://arxiv.org/abs/2207.04410)

</div>

## Project structure
```bash
├── README.md
├── comer               # model definition folder
│   └── losses          # Fisher loss (LayerWeightedFisherLoss)
├── convert2symLG       # official tool to convert latex to symLG format
├── lgeval              # official tool to compare symLGs in two folder
├── config.yaml         # config for CoMER hyperparameter
├── data.zip
├── eval_all.sh         # script to evaluate model on all CROHME test sets
├── example
│   ├── UN19_1041_em_595.bmp
│   └── example.ipynb   # HMER demo
├── lightning_logs      # training logs
│   └── version_0
│       ├── checkpoints
│       │   └── epoch=151-step=57151-val_ExpRate=0.6365.ckpt
│       ├── config.yaml
│       └── hparams.yaml
├── requirements.txt
├── scripts             # evaluation scripts
├── setup.cfg
├── setup.py
└── train.py
```

## Install dependencies   
```bash
cd CoMER
# install project   
conda create -y -n CoMER python=3.7
conda activate CoMER
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
 ```

## Training
Next, navigate to CoMER folder and run `train.py`. It may take **7~8** hours on **4** NVIDIA 2080Ti gpus using ddp.
```bash
# train CoMER(Fusion) model using 4 gpus and ddp
python train.py --config config.yaml  
```

You may change the `config.yaml` file to train different models
```yaml
# train BTTR(baseline) model
cross_coverage: false
self_coverage: false

# train CoMER(Self) model
cross_coverage: false
self_coverage: true

# train CoMER(Cross) model
cross_coverage: true
self_coverage: false

# train CoMER(Fusion) model
cross_coverage: true
self_coverage: true
```

For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1
# gpus: 4
# accelerator: ddp
```

## Layer-weighted Fisher loss (optional)

Training can add a **layer-weighted Fisher loss** on decoder hidden states, combined with the original bidirectional CE loss:

```
total_loss = CE + λ * Fisher        (only when epoch >= fisher_warmup_epoch)
```

Implementation lives in `comer/losses/fisher_loss.py` (`LayerWeightedFisherLoss`). Each decoder layer produces hidden states `[B, L, D]`; a learnable softmax over layers weights per-layer Fisher terms. Features are projected, L2-normalized, and scored with a Fisher criterion (within-class vs. between-class scatter).

### Teacher-forcing label alignment

Under teacher forcing, decoder input is `tgt` and targets are `out` (from `to_bi_tgt_out`). Fisher uses a full sequence `y` with `y[:, 1:] == out`:

```python
y = build_teacher_forcing_labels(tgt, out)  # [B, L + 1]
```

Hidden state at time `t` is paired with `out[:, t]`. No extra flip is needed for the bidirectional batch (`[2B, L]`): L2R and R2L rows each use their own `tgt` / `out`.

### Config (`config.yaml`)

```yaml
# fisher loss
use_fisher_loss: true              # set false to train with CE only
lambda_fisher: 0.005               # weight λ after warmup
fisher_warmup_epoch: 20            # CE-only for epochs 0..19; Fisher from epoch 20
fisher_proj_dim: 128               # projection dim for Fisher features
fisher_ignore_special_tokens: true # ignore PAD, SOS, EOS in Fisher (if false, only PAD)
learn_layer_weight: true           # learn softmax layer weights (if false, uniform 1/L)
```

### Training logs

When `use_fisher_loss: true`, Lightning logs:

- `train_ce_loss` — cross-entropy only
- `train_fisher_loss` — Fisher term (logged from `fisher_warmup_epoch` onward)
- `train_loss` — total optimized loss (`CE` or `CE + λ * Fisher`)
- `train_fisher_layer_weight_{i}` — softmax weight for decoder layer `i`

Validation still uses CE only (no Fisher at inference).

## Evaluation
Metrics used in validation during the training process is not accurate.

For accurate metrics reported in the paper, please use tools officially provided by CROHME 2019 oganizer:

A trained CoMER(Fusion) weight checkpoint has been saved in `lightning_logs/version_0`



```bash
perl --version  # make sure you have installed perl 5

unzip -q data.zip

# evaluation
# evaluate model in lightning_logs/version_0 on all CROHME test sets
# results will be printed in the screen and saved to lightning_logs/version_0 folder
bash eval_all.sh 0
```