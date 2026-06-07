import zipfile
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.utilities.distributed import rank_zero_info
from torch import FloatTensor, LongTensor

from comer.datamodule import Batch, vocab
from comer.losses import LayerWeightedFisherLoss, build_teacher_forcing_labels
from comer.model.comer import CoMER
from comer.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                               to_bi_tgt_out)

_PAD_IDX, _SOS_IDX, _EOS_IDX = 0, 1, 2


class LitCoMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
        # fisher loss
        use_fisher_loss: bool = False,
        lambda_fisher: float = 0.0,
        fisher_warmup_epoch: int = 0,
        fisher_proj_dim: Optional[int] = None,
        fisher_ignore_special_tokens: bool = True,
        learn_layer_weight: bool = True,
        pretrained_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.comer_model = CoMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.fisher_loss: Optional[LayerWeightedFisherLoss] = None
        if use_fisher_loss:
            ignore_index = (_PAD_IDX, _SOS_IDX, _EOS_IDX)
            if not fisher_ignore_special_tokens:
                ignore_index = (_PAD_IDX,)
            self.fisher_loss = LayerWeightedFisherLoss(
                num_layers=num_decoder_layers,
                feat_dim=d_model,
                proj_dim=fisher_proj_dim,
                ignore_index=ignore_index,
                learn_layer_weight=learn_layer_weight,
            )

        self.exprate_recorder = ExpRateRecorder()

    def on_train_epoch_start(self) -> None:
        device = self.device
        self._epoch_loss_sum = torch.zeros((), device=device)
        self._epoch_loss_count = torch.zeros((), device=device)

    def _accumulate_epoch_train_loss(self, loss: torch.Tensor) -> None:
        self._epoch_loss_sum = self._epoch_loss_sum + loss.detach()
        self._epoch_loss_count = self._epoch_loss_count + 1

    def get_epoch_train_loss(self) -> Optional[float]:
        if self._epoch_loss_count.item() == 0:
            return None
        loss_sum = self._epoch_loss_sum.clone()
        count = self._epoch_loss_count.clone()
        if self.trainer is not None and torch.distributed.is_initialized():
            if self.trainer.world_size > 1:
                torch.distributed.all_reduce(loss_sum)
                torch.distributed.all_reduce(count)
        return (loss_sum / count).item()

    def on_fit_start(self) -> None:
        path = self.hparams.pretrained_ckpt
        if not path:
            return

        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        incompatible = self.load_state_dict(state_dict, strict=False)
        rank_zero_info(
            "Loaded pretrained weights from %s (strict=False). "
            "missing=%d unexpected=%d",
            path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if incompatible.missing_keys:
            rank_zero_info("  missing: %s", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            rank_zero_info("  unexpected: %s", incompatible.unexpected_keys)

    def forward(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        tgt: LongTensor,
        return_all_features: bool = False,
    ) -> Union[FloatTensor, Tuple[FloatTensor, List[FloatTensor]]]:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]
        return_all_features : bool
            If True, also return per-layer decoder hidden states.

        Returns
        -------
        FloatTensor or (FloatTensor, List[FloatTensor])
            [2b, l, vocab_size], or logits plus layer features.
        """
        return self.comer_model(
            img, img_mask, tgt, return_all_features=return_all_features
        )

    def _fisher_active(self) -> bool:
        return (
            self.hparams.use_fisher_loss
            and self.current_epoch >= self.hparams.fisher_warmup_epoch
        )

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)

        if self._fisher_active():
            out_hat, layer_features = self(
                batch.imgs, batch.mask, tgt, return_all_features=True
            )
            ce = ce_loss(out_hat, out)
            target_labels = build_teacher_forcing_labels(tgt, out)
            fisher, layer_weights = self.fisher_loss(layer_features, target_labels)
            fisher_weighted = self.hparams.lambda_fisher * fisher
            loss = ce + fisher_weighted

            self.log("train_ce_loss", ce, on_step=False, on_epoch=True, sync_dist=True)
            self.log(
                "train_fisher_loss",
                fisher,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_fisher_weighted",
                fisher_weighted,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            for i, w in enumerate(layer_weights):
                self.log(
                    f"train_fisher_layer_weight_{i}",
                    w,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        else:
            out_hat = self(batch.imgs, batch.mask, tgt)
            loss = ce_loss(out_hat, out)
            self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            if self.hparams.use_fisher_loss:
                self.log(
                    "train_ce_loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        self._accumulate_epoch_train_loss(loss)
        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps]

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.comer_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            # Skip LR update when validation is skipped (no val_ExpRate logged).
            "strict": False,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
