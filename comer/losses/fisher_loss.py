from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Match comer.datamodule.vocab.CROHMEVocab special token ids.
_PAD_IDX, _SOS_IDX, _EOS_IDX = 0, 1, 2


def _zero_loss(z: Tensor) -> Tensor:
    """Scalar zero that stays connected to ``z`` for autograd."""
    return z.sum() * 0.0


def fisher_loss_single_layer(
    z: Tensor,
    labels: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Fisher criterion on L2-normalized token features.

    Minimizes within-class scatter relative to between-class separation of
    class means. Both ``z`` and ``labels`` must already be flattened to
  [N, D] and [N] respectively, with invalid positions removed.

    Parameters
    ----------
    z : Tensor
        [N, D], L2-normalized along the last dimension.
    labels : Tensor
        [N], token class indices.
    eps : float
        Stabilizer for the between-class term.

    Returns
    -------
    Tensor
        Scalar Fisher loss.
    """
    if z.numel() == 0 or labels.numel() == 0:
        return _zero_loss(z)

    classes = labels.unique()
    if classes.numel() < 2:
        return _zero_loss(z)

    global_mean = z.mean(dim=0)

    within = z.new_zeros(())
    class_means: List[Tensor] = []
    for c in classes:
        mask = labels == c
        z_c = z[mask]
        if z_c.numel() == 0:
            continue
        mu_c = z_c.mean(dim=0)
        class_means.append(mu_c)
        within = within + ((z_c - mu_c) ** 2).sum() / z_c.size(0)

    if len(class_means) < 2:
        return _zero_loss(z)

    within = within / len(class_means)

    class_means_t = torch.stack(class_means, dim=0)
    between = ((class_means_t - global_mean.unsqueeze(0)) ** 2).sum(dim=-1).mean()

    return within / (between + eps)


def build_teacher_forcing_labels(tgt: Tensor, out: Tensor) -> Tensor:
    """Build full sequence ``y`` so that ``y[:, 1:] == out``.

    CoMER uses decoder input ``tgt`` (``[B, L]``) and targets ``out`` (``[B, L]``).
    The full sequence is:

        y = concat(SOS column, out)   # shape [B, L + 1]
        y[:, 1:] == out

    Parameters
    ----------
    tgt : Tensor
        Decoder input tokens, ``[B, L]`` (column 0 must be SOS / direction start).
    out : Tensor
        Teacher-forced targets, ``[B, L]``.

    Returns
    -------
    Tensor
        ``[B, L + 1]`` label sequence for :func:`align_teacher_forcing`.
    """
    if tgt.size(1) != out.size(1):
        raise ValueError(
            f"tgt and out must share sequence length, got {tgt.size(1)} and {out.size(1)}"
        )
    return torch.cat([tgt[:, :1], out], dim=1)


def align_teacher_forcing(
    layer_features: List[Tensor],
    labels: Tensor,
) -> Tuple[List[Tensor], Tensor]:
    """Align decoder features to next-token targets under teacher forcing.

    Decoder input is ``labels[:, :-1]``; the prediction target is ``labels[:, 1:]``.
    Each hidden state at time ``t`` is paired with ``target[:, t]``.

    Parameters
    ----------
    layer_features : List[Tensor]
        Per-layer hiddens from the decoder, each ``[B, T_f, D]``.
    labels : Tensor
        Full sequence ``y`` of shape ``[B, T_y]`` where ``T_y = T_f + 1``.

    Returns
    -------
    aligned_features : List[Tensor]
        Each ``[B, T_f, D]``, trimmed so time length matches ``target``.
    target : Tensor
        ``labels[:, 1:]``, shape ``[B, T_f]``.
    """
    target = labels[:, 1:]
    tgt_len = target.size(1)

    aligned_features: List[Tensor] = []
    for feat in layer_features:
        if feat.size(1) < tgt_len:
            raise ValueError(
                f"Feature length {feat.size(1)} is shorter than target length {tgt_len}"
            )
        aligned_features.append(feat[:, :tgt_len, :])

    return aligned_features, target


class LayerWeightedFisherLoss(nn.Module):
    """Layer-weighted Fisher loss over decoder hidden states.

    A learnable softmax over decoder layers combines per-layer Fisher losses
    computed on projected, L2-normalized token features.

    Parameters
    ----------
    num_layers : int
        Number of decoder layers in ``layer_features``.
    feat_dim : int
        Hidden dimension ``D`` of each layer feature tensor.
    proj_dim : Optional[int]
        Output dimension of the projection head. If ``None``, features are
        used without an extra linear map (``proj`` is ``Identity``).
    ignore_index : Optional[Sequence[int]]
        Token indices excluded from the loss. Defaults to PAD, SOS, and EOS.
    eps : float
        Stabilizer passed to :func:`fisher_loss_single_layer`.
    """

    def __init__(
        self,
        num_layers: int,
        feat_dim: int,
        proj_dim: Optional[int] = None,
        ignore_index: Optional[Sequence[int]] = None,
        learn_layer_weight: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.learn_layer_weight = learn_layer_weight
        self.num_layers = num_layers

        if ignore_index is None:
            ignore_index = (_PAD_IDX, _SOS_IDX, _EOS_IDX)
        self.register_buffer(
            "ignore_index",
            torch.tensor(list(ignore_index), dtype=torch.long),
        )

        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        if not learn_layer_weight:
            self.layer_logits.requires_grad = False

        out_dim = proj_dim if proj_dim is not None else feat_dim
        if proj_dim is not None:
            self.proj = nn.Linear(feat_dim, proj_dim, bias=False)
        else:
            self.proj = nn.Identity()

        self._out_dim = out_dim

    def _mask_valid(
        self, feat: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Flatten and drop ignored token positions."""
        z = rearrange_flat(feat)
        y = labels.reshape(-1)

        valid = torch.ones_like(y, dtype=torch.bool)
        for idx in self.ignore_index.tolist():
            valid &= y != idx

        return z[valid], y[valid]

    def forward(
        self,
        layer_features: List[Tensor],
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute layer-weighted Fisher loss.

        Parameters
        ----------
        layer_features : List[Tensor]
            Per-layer decoder hiddens, each ``[B, L, D]``.
        labels : Tensor
            Full teacher-forcing sequence ``y`` of shape ``[B, L + 1]`` so that
            ``y[:, 1:]`` are the prediction targets. Use
            :func:`build_teacher_forcing_labels` with CoMER ``tgt`` and ``out``.

        Returns
        -------
        fisher_loss : Tensor
            Scalar combined loss.
        layer_weights : Tensor
            Softmax weights over layers, shape ``[num_layers]``.
        """
        if len(layer_features) != self.layer_logits.numel():
            raise ValueError(
                f"Expected {self.layer_logits.numel()} layer features, "
                f"got {len(layer_features)}"
            )

        layer_features, target = align_teacher_forcing(layer_features, labels)

        if self.learn_layer_weight:
            weights = torch.softmax(self.layer_logits, dim=0)
        else:
            weights = layer_features[0].new_full(
                (self.num_layers,), 1.0 / self.num_layers
            )

        loss = torch.zeros((), device=labels.device, dtype=layer_features[0].dtype)
        for w, feat in zip(weights, layer_features):
            z, y = self._mask_valid(feat, target)
            z = self.proj(z)
            z = F.normalize(z, dim=-1, eps=1e-6)
            loss = loss + w * fisher_loss_single_layer(z, y, eps=self.eps)

        return loss, weights


def rearrange_flat(feat: Tensor) -> Tensor:
    """[B, L, D] -> [B * L, D]."""
    return feat.reshape(feat.size(0) * feat.size(1), feat.size(2))
