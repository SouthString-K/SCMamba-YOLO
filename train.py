from __future__ import annotations

import argparse
import os

from ultralytics import YOLO
from ultralytics.nn.modules.enhance_front import configure_enhance_runtime


def resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="COCO/coco.yaml", help="dataset yaml path")
    parser.add_argument(
        "--config",
        type=str,
        default="ultralytics/cfg/models/scmamba-yolo/SCMamba-YOLO-T-Enhance.yaml",
        help="model yaml path",
    )
    parser.add_argument("--weights", type=str, default=None, help="optional checkpoint (.pt) to warm-start detector")
    parser.add_argument("--enhance_weights", type=str, default=None, help="optional InteractNet checkpoint (.pth)")
    parser.add_argument(
        "--enhance_freeze",
        action="store_true",
        help="freeze the enhancement front-end during detector training",
    )
    parser.add_argument("--batch_size", type=int, default=-1, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="input size (pixels)")
    parser.add_argument("--task", default="train", help="train, val, or test")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--freeze", type=int, default=None, help="freeze first n detector layers during training")
    parser.add_argument("--optimizer", default="SGD", help="SGD, Adam, AdamW")
    parser.add_argument("--amp", action="store_true", help="enable amp")
    parser.add_argument("--project", default="tf-logs", help="save to project/name")
    parser.add_argument("--name", default="scmambayolo_train", help="save to project/name")
    parser.add_argument("--resume", type=str, default="tf-logs/EMA/weights/last.pt", help="resume from checkpoint")
    return parser.parse_args()


def build_common_args(opt):
    args = {
        "data": resolve_path(opt.data),
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": resolve_path(opt.project),
        "name": opt.name,
    }
    if opt.freeze is not None:
        args["freeze"] = opt.freeze
    return args


def uses_enhance_config(config_path: str | None) -> bool:
    if not config_path:
        return False
    return PathLike(config_path).name.endswith("-Enhance.yaml")


class PathLike(str):
    @property
    def name(self):
        return os.path.basename(self)


def enhancement_weights_error(config_path: str) -> str:
    return (
        f"The selected config '{config_path}' enables the enhancement front-end, so "
        "`--enhance_weights` is required for official training/validation.\n"
        "How to obtain enhancement weights:\n"
        "1. Go to Enhancement-main/.\n"
        "2. Train InteractNet with: python src/train.py --dataset-name UIEB --train-dir <train_dir/> --valid-dir <val_dir/> --ckpt-save-path <save_dir>\n"
        "3. Use the generated checkpoint (.pt/.pth) as --enhance_weights.\n"
        "If you only want to inspect the architecture, you may instantiate the enhancement module manually, "
        "but the released train/val pipeline requires pretrained enhancement weights."
    )


if __name__ == "__main__":
    opt = parse_opt()
    task = opt.task.lower()

    config_path = resolve_path(opt.config)
    resume_path = resolve_path(opt.resume)
    weights_path = resolve_path(opt.weights)
    enhance_weights_path = resolve_path(opt.enhance_weights)

    configure_enhance_runtime(
        enable=True,
        weights=enhance_weights_path,
        freeze=opt.enhance_freeze,
    )

    args = build_common_args(opt)

    if task == "train":
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming training from {resume_path} ...")
            model = YOLO(resume_path)
            model.train(resume=True, **args)
        else:
            if uses_enhance_config(config_path) and not enhance_weights_path:
                raise ValueError(enhancement_weights_error(config_path))
            print(f"Starting training from config {config_path} ...")
            model = YOLO(config_path)
            if weights_path:
                print(f"Loading detector warm-start weights from {weights_path} ...")
                model.load(weights_path)
            model.train(**args)

    elif task == "val":
        if not weights_path and uses_enhance_config(config_path) and not enhance_weights_path:
            raise ValueError(enhancement_weights_error(config_path))
        model = YOLO(weights_path or config_path)
        model.val(**args)

    elif task == "test":
        if not weights_path and uses_enhance_config(config_path) and not enhance_weights_path:
            raise ValueError(enhancement_weights_error(config_path))
        model = YOLO(weights_path or config_path)
        model.val(split="test", **args)

    else:
        raise ValueError(f"Unsupported task: {opt.task}. Use train, val, or test.")
