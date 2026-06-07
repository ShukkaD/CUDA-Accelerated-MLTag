from pathlib import Path
import os
import random
import numpy as np
import torch
from typing import Any

import albumentations as A
from ultralytics import YOLO
import ultralytics.data.augment as augment
from ultralytics.utils import LOGGER, colorstr
from ultralytics.data.augment import BaseTransform


main_dir = Path(__file__).resolve().parent.parent
train_dir = main_dir / "TrainingRuns"

DATA_YAML = main_dir / "apriltagsDataset" / "data.yaml"
LABEL_DIRS = [
    DATA_YAML.parent / "train" / "labels", 
    DATA_YAML.parent / "valid" / "labels", 
    DATA_YAML.parent / "test" / "labels"
]
IMAGE_SIZE = 640
TRAIN_BATCH_SIZE = 16

model_path = main_dir / "yolo26n.pt"

def segment_to_box(line: str) -> str:
    parts = line.split()
    class_id = parts[0]
    points = list(map(float, parts[1:]))
    xs = points[0::2]
    ys = points[1::2]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return f"{class_id} {x_center:.16g} {y_center:.16g} {width:.16g} {height:.16g}"


def normalize_label_files() -> None:
    for label_dir in LABEL_DIRS:
        for label_file in label_dir.glob("*.txt"):
            lines = label_file.read_text().splitlines()
            normalized_lines = []
            changed = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) > 5:
                    normalized_lines.append(segment_to_box(stripped))
                    changed = True
                else:
                    normalized_lines.append(stripped)
            if changed:
                label_file.write_text("\n".join(normalized_lines) + ("\n" if normalized_lines else ""))


def clear_dataset_caches() -> None:
    for cache_file in (
        DATA_YAML.parent / "train" / "labels.cache",
        DATA_YAML.parent / "valid" / "labels.cache",
        DATA_YAML.parent / "test" / "labels.cache",
    ):
        cache_file.unlink(missing_ok=True)

# Custom Augmentation Pipeline
CUSTOM_TRANSFORMS = [
    
    A.HorizontalFlip(p=0.5),

    #Real life deployment is always grayscale
    A.ToGray(p=1.0),

    A.Affine(scale=(0.5, 2), translate_percent=(-0.1, 0.1), rotate=(-25, 25), shear=(-15, 15), p=0.75),
    A.Perspective(p=0.75),
    A.Illumination(p=1.0),
    A.RandomBrightnessContrast(brightness_range=(-0.35, 0.25), contrast_range=(-0.3, 0.3), p=1.0),
    A.RandomGamma((75, 160)),
    A.LensFlare(intensity_range=(0.075, 0.325)),
    A.CLAHE((1, 6)),
    A.ISONoise(intensity_range=(0.075, 0.175), p=0.3),
    A.MotionBlur((3, 9)),
    A.Defocus(radius_range=(1,2), alias_blur_range=(0.1, 0.4), p=0.3),
    A.ImageCompression(quality_range=(40, 80), p=0.3),
    A.Downscale((0.65, 0.9), p=0.3),
    A.RandomShadow(shadow_dimension=4, shadow_intensity_range=(0.2, 0.55), p=0.3)

]


class SafeCustomAlbumentations(BaseTransform):
    """
    Standard training patched Albumentations pipeline
    Fixes albumentationsx (coord_format) and OpenCV read-only buffers
    """
    def __init__(self, p: float = 1.0, transforms: list | None = None, *args, **kwargs) -> None:
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  
            import albumentations as A

            # List of possible spatial transforms natively extracted from Ultralytics
            spatial_transforms = {
                "Affine", "BBoxSafeRandomCrop", "CenterCrop", "CoarseDropout", "Crop",
                "CropAndPad", "CropNonEmptyMaskIfExists", "D4", "ElasticTransform",
                "Flip", "GridDistortion", "GridDropout", "HorizontalFlip", "Lambda",
                "LongestMaxSize", "MaskDropout", "MixUp", "Morphological", "NoOp",
                "OpticalDistortion", "PadIfNeeded", "Perspective", "PiecewiseAffine",
                "PixelDropout", "RandomCrop", "RandomCropFromBorders", "RandomGridShuffle",
                "RandomResizedCrop", "RandomRotate90", "RandomScale", "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop", "Resize", "Rotate", "SafeRotate", "ShiftScaleRotate",
                "SmallestMaxSize", "Transpose", "VerticalFlip", "XYMasking",
            }

            # If transforms isn't provided natively by Ultralytics, use your global CUSTOM_TRANSFORMS
            T = transforms if transforms is not None else CUSTOM_TRANSFORMS

            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            
            # Use 'coord_format' for albumentationsx instead of 'format'
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(coord_format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            
            if hasattr(self.transform, "set_random_seed"):
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
            
        except ImportError:  
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.transform is None or random.random() > self.p:
            return labels

        im = labels.get("img")
        if im is None or im.shape[2] != 3: 
            return labels

        if self.contains_spatial:
            cls = labels.get("cls", np.zeros((0, 1)))
            
            if len(cls):
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
            else:
                bboxes = np.zeros((0, 4))

            # Flatten array instead of squeezing for consistent lists
            class_labels = cls.flatten().tolist() if len(cls) else []

            # Apply transform normally without warning suppression
            new = self.transform(
                image=im,
                bboxes=bboxes,
                class_labels=class_labels,
            )

            # --- PATCH: Force a deep copy to guarantee writeable memory for OpenCV ---
            labels["img"] = new["image"].copy()
            
            if len(cls):
                if len(new["bboxes"]):
                    labels["instances"].update(bboxes=np.array(new["bboxes"], dtype=np.float32))
                    labels["instances"].denormalize(*im.shape[:2][::-1])
                    labels["cls"] = np.array(new["class_labels"], dtype=np.float32)[..., np.newaxis]
                else:
                    labels["cls"] = np.zeros((0, 1), dtype=np.float32)
                    labels["instances"].update(bboxes=np.zeros((0, 4), dtype=np.float32))
        else:
            new = self.transform(image=im)
            
            # Force a deep copy to guarantee writeable memory for OpenCV
            labels["img"] = new["image"].copy()

        return labels


def main():
    
    os.chdir(main_dir)
    
    normalize_label_files()
    clear_dataset_caches()
    model = YOLO(model_path) # Ensure this model file exists natively in your path
    
    # Inject the dynamically built class directly into Ultralytics
    augment.Albumentations = SafeCustomAlbumentations

    model.train(
        data=str(DATA_YAML),
        epochs=350,
        imgsz=IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        device=0,
        optimizer="MuSGD",
        cutmix=0.25,
        project=str(train_dir),
        augmentations=CUSTOM_TRANSFORMS
    )

if __name__ == "__main__":
    main()