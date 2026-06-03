from pathlib import Path

import albumentations as A

from ultralytics import YOLO

main_dir = Path(__file__).resolve().parent.parent

DATA_YAML = main_dir / "apriltagsDataset" / "data.yaml"
LABEL_DIRS = [DATA_YAML.parent / "train" / "labels", DATA_YAML.parent / "valid" / "labels", DATA_YAML.parent / "test" / "labels"]
IMAGE_SIZE = 640
TRAIN_BATCH_SIZE = 16

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

custom_transforms = [
    A.HorizontalFlip(p=0.5),
    A.ConstrainedCoarseDropout((1,3)),
    A.CoarseDropout((1,3)),
    A.ToGray(p=0.75),
    A.Affine(scale=(0.5, 2), translate_percent=(-0.1, 0.1), rotate=(-15, 15), shear=(-15, 15), p=0.75),
    A.Perspective(scale=(0.05, 0.15), p=0.75),
    A.Illumination(),
    A.RandomBrightnessContrast(),
    A.RandomGamma((40, 160)),
    A.LensFlare(),
    A.CLAHE((1, 6)),
    A.ISONoise(p=0.3),
    A.MotionBlur((3, 11)),
    A.Defocus(radius_range=(1,3), alias_blur_range=(0.1, 0.4), p=0.3),
    A.ImageCompression(quality_range=(25, 75)),
    A.Downscale((0.65, 0.9)),
    A.RandomShadow(shadow_dimension=4, shadow_intensity_range=(0.2, 0.55), p=0.3)
]

def main():
    normalize_label_files()
    clear_dataset_caches()
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=350,
        imgsz=IMAGE_SIZE,
        batch=TRAIN_BATCH_SIZE,
        device=0,
        optimizer="auto",
        cutmix=0.25,
        augmentations=custom_transforms
    )

if __name__ == "__main__":
    main()
