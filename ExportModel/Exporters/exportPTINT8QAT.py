import os

from modelopt_compat import patch_modelopt_environment

patch_modelopt_environment()

import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer


from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Albumentations

import logging
import copy
import albumentations as A
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from multiprocessing import freeze_support
from pathlib import Path
from types import SimpleNamespace

main_dir = Path(__file__).resolve().parent.parent

aprilTagsDatasetPath = os.path.join(main_dir, "apriltagsDataset")
dataYAML = os.path.join(aprilTagsDatasetPath, 'data.yaml')
labelDIRS = [Path(dataYAML).parent / "train" / "labels", Path(dataYAML).parent / "valid" / "labels", Path(dataYAML).parent / "test" / "labels"]
imgsize = 640

batchsz_train = 16
batchsz_calib = 1
batchsz_val = 1
batchsz_export = 1
minimum_calibration_images = 300

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logging.getLogger("modelopt").setLevel(logging.DEBUG)
logging.getLogger("modelopt.torch.quantization").setLevel(logging.DEBUG)

base_model_path = os.path.join(main_dir, 'yolo26nApriltag.pt')

yolo26 = YOLO(base_model_path)

# Base YOLO model before quantization.
torch_model = yolo26.model
torch_model = torch_model.to(device).eval()

config = mtq.INT8_DEFAULT_CFG


def saveExport(model, path):
    """Save the raw ModelOpt quantized checkpoint without leaving the device."""
    path = Path(path)
    if path.suffix.lower() in (".pt", ".pth"):
        checkpoint_path = path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = path / "model.pth"

    # Save using ModelOpt's saver directly from the current device
    mto.save(model, checkpoint_path)

    return checkpoint_path


def ensure_model_hyperparameters(model):
    if isinstance(model.args, dict):
        model.args = SimpleNamespace(**model.args)

    for key, value in DEFAULT_CFG_DICT.items():
        if not hasattr(model.args, key):
            setattr(model.args, key, value)


def freeze_quantizers_for_qat(model):
    for module in model.modules():
        if not isinstance(module, TensorQuantizer):
            continue

        if hasattr(module, "disable_calib"):
            module.disable_calib()

        if hasattr(module, "enable_quant"):
            module.enable_quant()


def ensure_trainable_parameters_fp32(model):
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()


def force_optimizer_params_fp32(optimizer):
    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group["params"]:
            if parameter is not None and parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()


class NoFuseValidationWrapper(torch.nn.Module):
    """Hide .fuse() from Ultralytics validation so ModelOpt quantized models stay intact."""

    def __init__(self, wrapped_model):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.stride = wrapped_model.stride
        self.names = wrapped_model.names
        self.yaml = getattr(wrapped_model, "yaml", {})
        head = wrapped_model.model[-1]
        self.end2end = getattr(head, "end2end", getattr(wrapped_model, "end2end", False))

    def set_head_attr(self, **kwargs):
        if hasattr(self.wrapped_model, "set_head_attr"):
            self.wrapped_model.set_head_attr(**kwargs)
        elif hasattr(self.wrapped_model.model[-1], "set_head_attr"):
            self.wrapped_model.model[-1].set_head_attr(**kwargs)

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)


def validate(model, val_loader, device=device):
    model.eval()
    total = 0.0
    count = 0
    use_amp = device == "cuda"
    amp_dtype = torch.float16
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["img"].to(device).float() / 255.0
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            try:
                if use_amp:
                    with torch.autocast(device_type=device, dtype=amp_dtype):
                        preds = model(imgs)
                        loss, _ = model.loss(batch, preds)
                else:
                    preds = model(imgs)
                    loss, _ = model.loss(batch, preds)
            except Exception as amp_error:
                logging.warning(
                    "AMP validation failed on %s; falling back to FP32 for the rest of validation: %s",
                    device,
                    amp_error,
                )
                use_amp = False
                preds = model(imgs)
                loss, _ = model.loss(batch, preds)
            if getattr(loss, "ndim", 0):
                loss = loss.sum()
            total += float(loss.detach())
            count += 1
    return total / max(1, count)


def compute_exact_box_map_from_model(
    live_model,
    data_yaml=dataYAML,
    imgsz=imgsize,
    batch=batchsz_val,
):
    """Return the exact Ultralytics mAP50-95 value from the live model.

    Ultralytics computes detection mAP50-95 by accumulating per-image stats in DetectionValidator,
    then calling `DetMetrics.process()`, which updates `metrics.box`. The documented accessor is `metrics.box.map`.
    """
    was_training = live_model.training
    validation_model = None
    try:
        validation_model = NoFuseValidationWrapper(copy.deepcopy(live_model)).to(device).eval()
        validator_args = dict(
            model=base_model_path,
            data=data_yaml,
            imgsz=imgsz,
            batch=batch,
            task="detect",
            split="val",
            device=device,
            half=False,
            conf=0.001,
            iou=0.7,
            verbose=False,
            plots=False,
            save_json=False,
            save_txt=False,
            save_conf=False,
            augment=False,
            workers=0,
            single_cls=False,
            agnostic_nms=False,
            max_det=300
        )
        validator = DetectionValidator(args=validator_args)
        stats = validator(model=validation_model)
        map50_95 = stats.get("metrics/mAP50-95(B)")
        if map50_95 is None and getattr(validator.metrics, "box", None) is not None:
            map50_95 = validator.metrics.box.map
        return float(map50_95)
    except Exception as exc:
        logging.warning(
            "Exact mAP50-95 validation failed on the live model: %s",
            exc,
        )
        return None
    finally:
        del validation_model
        if device == "cuda":
            torch.cuda.empty_cache()
        if was_training:
            live_model.train()


def train_qat(
    model,
    train_loader,
    epochs,
    val_loader=None,
    save_dir=None,
):

    model.train()
    model.requires_grad_(True)

    ensure_model_hyperparameters(model)
    model.criterion = model.init_criterion()

    save_dir = Path(save_dir) if save_dir is not None else Path(os.path.join(main_dir, "PTCheckpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    use_amp = device == "cuda"
    amp_dtype = torch.float16

    ensure_trainable_parameters_fp32(model)
    freeze_quantizers_for_qat(model)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=5e-5,
        weight_decay=1e-4,
    )

    force_optimizer_params_fp32(optimizer)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=5e-7,
    )

    use_ema = True
    ema_model = None
    if use_ema:
        try:
            ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))
            ema_model = ema_model.to(device)
        except Exception as exc:
            logging.warning("EMA init failed; continuing without EMA: %s", exc)
            ema_model = None

    scaler = torch.amp.GradScaler(device=device) if use_amp else None
    amp_failed = False

    best_val = float("inf")
    best_validation_map = float("-inf")

    for epoch in range(epochs):

        model.train()
        freeze_quantizers_for_qat(model)
        force_optimizer_params_fp32(optimizer)

        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):

            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            imgs = batch["img"].float() / 255.0
            optimizer.zero_grad(set_to_none=True)

            current_use_amp = use_amp and not amp_failed
            if current_use_amp:
                force_optimizer_params_fp32(optimizer)

            try:
                if current_use_amp:
                    with torch.autocast(device_type=device, dtype=amp_dtype):
                        preds = model(imgs)
                        loss, _ = model.loss(batch, preds)
                else:
                    preds = model(imgs)
                    loss, _ = model.loss(batch, preds)
            except Exception as amp_error:
                logging.warning(
                    "AMP training failed on %s at epoch %d batch %d; falling back to FP32 for the rest of training: %s",
                    device,
                    epoch + 1,
                    batch_idx + 1,
                    amp_error,
                )
                amp_failed = True
                preds = model(imgs)
                loss, _ = model.loss(batch, preds)

            if getattr(loss, "ndim", 0):
                loss = loss.sum()

            if current_use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if current_use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if ema_model is not None:
                ema_model.update_parameters(model)

            running_loss += float(loss.item())

            log_interval = max(1, len(train_loader) // 20)
            if batch_idx == 0 or batch_idx + 1 == len(train_loader) or (batch_idx + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} "
                    f"Batch {batch_idx+1}/{len(train_loader)} "
                    f"Loss {loss.item()}"
                )

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.6f}")

        eval_model = ema_model.module if ema_model is not None else model

        val_loss = None
        validation_map = None

        if val_loader is not None:
            eval_model.eval()
            with torch.no_grad():
                val_loss = validate(eval_model, val_loader)
                print(f"Epoch {epoch+1} Validation Loss: {val_loss}")

        # Compute exact mAP50-95 before saveExport mutates quantizer state.
        if val_loader is not None:
            validation_map = compute_exact_box_map_from_model(eval_model)
            if validation_map is not None:
                print(f"Epoch {epoch+1} mAP@0.5:0.95: {validation_map}")

        # Save LAST after validation while the live model is still intact.
        last_path = save_dir / "last"
        orig_state = {
            k: v.detach().float().clone()
            for k, v in eval_model.state_dict().items()
        }

        # Save export (ModelOpt) and produce a YOLO-wrapped checkpoint when possible.
        saveExport(eval_model, last_path)

        with torch.inference_mode():
            eval_model.load_state_dict(orig_state, strict=False)
        freeze_quantizers_for_qat(eval_model)
        if eval_model is not model:
            freeze_quantizers_for_qat(model)

        saved_best = False

        if validation_map is not None:
            if validation_map > best_validation_map:
                best_validation_map = validation_map
                best_path = save_dir / "best"
                print(f"New Best mAP@0.5:0.95: {best_validation_map}")

                orig_state = {
                    k: v.detach().float().clone()
                    for k, v in eval_model.state_dict().items()
                }

                saveExport(eval_model, best_path)

                with torch.inference_mode():
                    eval_model.load_state_dict(orig_state, strict=False)
                freeze_quantizers_for_qat(eval_model)
                if eval_model is not model:
                    freeze_quantizers_for_qat(model)
                saved_best = True

        elif val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_path = save_dir / "best"
            print(f"New Lowest Loss: {best_val}")

            orig_state = {
                k: v.detach().float().clone()
                for k, v in eval_model.state_dict().items()
            }

            saveExport(eval_model, best_path)

            with torch.inference_mode():
                eval_model.load_state_dict(orig_state, strict=False)
            freeze_quantizers_for_qat(eval_model)
            if eval_model is not model:
                freeze_quantizers_for_qat(model)
            saved_best = True

        if saved_best:
            print(f"Saved new best model to {best_path}")

        scheduler.step()
        print(f"Epoch {epoch+1} LR: {scheduler.get_last_lr()[0]:.2e}")

        print("-" * 60)


    print("QAT training complete.")
    print("-" * 60)
    print(f"Best mAP@0.5:0.95: {best_validation_map}")



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
    for label_dir in labelDIRS:

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

class QATYOLODataset(YOLODataset):
    
    def __init__(self, qat_transforms, *args, **kwargs):
        # Store custom transforms before calling super() so they exist when build_transforms fires
        self.qat_custom_transforms = qat_transforms
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        # Let YOLO build its native transform pipeline first
        transforms = super().build_transforms(hyp)
        
        # Cleanly swap out the Albumentations wrapper natively during initialization
        if hasattr(self, "albumentations") and self.qat_custom_transforms:
            qat_alb_wrapper = Albumentations(p=1.0)
            qat_alb_wrapper.transform = A.Compose(
                self.qat_custom_transforms,
                bbox_params=A.BboxParams(coord_format="yolo", label_fields=["apriltag"]),
            )
            self.albumentations = qat_alb_wrapper
            
        return transforms


def main():
    
    normalize_label_files()
    data = check_det_dataset(dataYAML)

    def forward_loop(model):
        model.eval()

        with torch.no_grad():
            for batch_index, batch in enumerate(calib_loader, start=1):
                # ModelOpt calibration runs the model with FP32 weights here, so keep inputs FP32.
                images = batch["img"].to(device, non_blocking=True).float() / 255.0
                logging.info(
                    "Calibrating batch %d/%d",
                    batch_index,
                    len(calib_loader),
                )
                _ = model(images)

    cfg = SimpleNamespace(**DEFAULT_CFG_DICT)
    # Safely copy hyperparameters from model.args whether dict or SimpleNamespace
    args_obj = getattr(torch_model, "args", None)
    if isinstance(args_obj, dict):
        items = args_obj.items()
    elif isinstance(args_obj, SimpleNamespace):
        items = vars(args_obj).items()
    else:
        try:
            items = dict(args_obj).items()
        except Exception:
            items = []

    for key, value in items:
        setattr(cfg, key, value)
    cfg.imgsz = imgsize
    cfg.cache = False
    cfg.single_cls = False
    cfg.task = getattr(cfg, "task", "detect")
    cfg.classes = getattr(cfg, "classes", None)
    cfg.fraction = getattr(cfg, "fraction", 1.0)
    cfg.mosaic = 0.85
    cfg.cutmix = 0.25
    cfg.close_mosaic = 5
    cfg.hsv_h = 0.0
    cfg.hsv_s = 0.0
    cfg.hsv_v = 0.3

    qat_custom_transforms = [
        A.HorizontalFlip(p=0.5),
        A.ToGray(p=0.75),
        A.Affine(scale=(0.65, 1.85), translate_percent=(-0.075, 0.075), rotate=(-12.5, 12.5), shear=(-12.5, 12.5), p=0.75),
        A.Perspective(scale=(0.05, 0.15), p=0.75),
        A.RandomBrightnessContrast(),
        A.Illumination(),
        A.RandomBrightnessContrast(),
        A.RandomGamma((40, 160)),
        A.LensFlare(),
        A.CLAHE(),
        A.ISONoise(p=0.3),
        A.MotionBlur((3, 8)),
        A.Defocus(radius_range=(1,3), alias_blur_range=(0.1, 0.4), p=0.3),
        A.ImageCompression(quality_range=(25, 75)),
        A.Downscale((0.75, 0.95))
    ]

    calibration_paths = [
        data[split]
        for split in ("train", "val", "test")
        if data.get(split)
    ]
    train_paths = [
        data[split]
        for split in ("train", "test")
        if data.get(split)
    ]

    aprilTagsDataset = build_yolo_dataset(
        cfg=cfg,
        img_path=calibration_paths,
        batch=batchsz_calib,
        data=data,
        mode="val",
        stride=32,
    )

    if len(aprilTagsDataset) < minimum_calibration_images:
        logging.warning(
            "Calibration set has %d images; target at least %d diverse frames for stable INT8 ranges.",
            len(aprilTagsDataset),
            minimum_calibration_images,
        )

    aprilTagsDatasetTrain = QATYOLODataset(
        qat_transforms=qat_custom_transforms,
        img_path=train_paths,
        imgsz=cfg.imgsz,
        batch_size=batchsz_train,
        augment=True,
        hyp=cfg,
        stride=32,
        data=data,
        task="detect"
    )

    aprilTagsDatasetVal = build_yolo_dataset(
        cfg=cfg,
        img_path=data["val"],
        batch=batchsz_val,
        data=data,
        mode="val",
        stride=32,
    )


    is_windows = os.name == "nt"
    calib_workers = 0 if is_windows else 4
    train_workers = 0 if is_windows else 4
    val_workers = 0 if is_windows else 2

    calib_loader = DataLoader(
        aprilTagsDataset,
        batch_size=batchsz_calib,
        shuffle=False,
        num_workers=calib_workers,
        pin_memory=device == "cuda",
        persistent_workers=calib_workers > 0,
        collate_fn=aprilTagsDataset.collate_fn,
    )

    train_loader = DataLoader(
        aprilTagsDatasetTrain,
        batch_size=batchsz_train,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=device == "cuda",
        persistent_workers=train_workers > 0,
        collate_fn=aprilTagsDatasetTrain.collate_fn,
    )

    val_loader = DataLoader(
        aprilTagsDatasetVal,
        batch_size=batchsz_val,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=device == "cuda",
        persistent_workers=val_workers > 0,
        collate_fn=aprilTagsDatasetVal.collate_fn,
    )


    quantized_model = mtq.quantize(
        torch_model,
        config,
        forward_loop,
    )

    quantized_model = quantized_model.to(device)

    # Print summary while quantizers are still present
    mtq.print_quant_summary(quantized_model)

    train_qat(model=quantized_model, train_loader=train_loader, epochs=35, val_loader=val_loader)

def exportINT8():
    freeze_support()
    main()

if __name__ == "__main__":
    exportINT8()
