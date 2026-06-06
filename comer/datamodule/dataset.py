from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import torchvision.transforms as tr
from PIL import Image
from torch.utils.data.dataset import Dataset

from .transforms import ScaleAugmentation, ScaleToLimitRange

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024

MAX_SIZE = 32e4
MAX_FORMULA_LEN = 200
DEFAULT_MAX_WIDTH = 400
DEFAULT_MAX_HEIGHT = 200


@dataclass(frozen=True)
class SampleRecord:
    img_name: str
    img_path: Path
    formula: Tuple[str, ...]
    width: int
    height: int
    area: int


def _parse_caption_file(caption_path: Path) -> List[Tuple[str, List[str]]]:
    samples = []
    for line in caption_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        samples.append((parts[0], parts[1:]))
    return samples


def load_sample_index(data_dir: Union[str, Path]) -> List[SampleRecord]:
    """Index caption.txt and image paths without loading pixel data."""
    root = Path(data_dir)
    caption_path = root / "caption.txt"
    img_dir = root / "img"
    if not caption_path.is_file():
        raise FileNotFoundError(f"caption.txt not found under {root}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"img/ not found under {root}")

    records: List[SampleRecord] = []
    for img_name, formula in _parse_caption_file(caption_path):
        img_path = img_dir / f"{img_name}.bmp"
        if not img_path.is_file():
            continue
        with Image.open(img_path) as img:
            width, height = img.size
        records.append(
            SampleRecord(
                img_name=img_name,
                img_path=img_path.resolve(),
                formula=tuple(formula),
                width=width,
                height=height,
                area=width * height,
            )
        )

    print(f"Indexed {len(records)} samples from: {root}")
    return records


def build_batch_indices(
    samples: List[SampleRecord],
    batch_size: int,
    batch_imagesize: float = MAX_SIZE,
    maxlen: int = MAX_FORMULA_LEN,
    max_imagesize: float = MAX_SIZE,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
) -> List[List[int]]:
    """Group sample indices into batches by sorted image area (same logic as before)."""
    order = sorted(range(len(samples)), key=lambda i: samples[i].area)

    batch_indices: List[List[int]] = []
    current: List[int] = []
    biggest_image_size = 0
    i = 0

    for idx in order:
        sample = samples[idx]
        size = sample.area

        if len(sample.formula) > maxlen:
            print("sentence", idx, "length bigger than", maxlen, "ignore")
            continue
        if size > max_imagesize:
            print(
                f"image: {sample.img_name} size bigger than {max_imagesize}, ignore"
            )
            continue
        if sample.width > max_width or sample.height > max_height:
            print(
                f"image: {sample.img_name} size: {sample.width} x {sample.height} "
                f"exceeds max {max_width} x {max_height}, ignore"
            )
            continue

        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)

        if batch_image_size > batch_imagesize or i == batch_size:
            if current:
                batch_indices.append(current)
            current = [idx]
            biggest_image_size = size
            i = 1
        else:
            current.append(idx)
            i += 1

    if current:
        batch_indices.append(current)

    print(f"total {len(batch_indices)} batch groups loaded")
    return batch_indices


def build_lazy_dataset(
    data_dir: Union[str, Path],
    batch_size: int,
) -> Tuple[List[SampleRecord], List[List[int]]]:
    samples = load_sample_index(data_dir)
    batch_indices = build_batch_indices(samples, batch_size)
    return samples, batch_indices


class CROHMEDataset(Dataset):
    """Lazy dataset: each item is one pre-grouped batch loaded on demand."""

    def __init__(
        self,
        samples: List[SampleRecord],
        batch_indices: List[List[int]],
        is_train: bool,
        scale_aug: bool,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.batch_indices = batch_indices

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        indices = self.batch_indices[idx]
        fnames = []
        imgs = []
        captions = []
        for sample_idx in indices:
            sample = self.samples[sample_idx]
            with Image.open(sample.img_path) as img:
                img = img.copy()
            fnames.append(sample.img_name)
            imgs.append(self.transform(img))
            captions.append(list(sample.formula))
        return fnames, imgs, captions

    def __len__(self):
        return len(self.batch_indices)
