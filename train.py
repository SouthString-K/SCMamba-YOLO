from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath(".")


def resolve_path(path: str | None) -> str | None:
    """Return absolute path while keeping already-absolute inputs unchanged."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="COCO/coco.yaml", help="dataset.yaml path")
    parser.add_argument(
        "--config",
        type=str,
        default="ultralytics/cfg/models/scmamba-yolo/SCMamba-YOLO-T.yaml",
        help="model yaml path",
    )
    parser.add_argument("--weights", type=str, default=None, help="optional checkpoint (.pt) to load before training")
    parser.add_argument("--batch_size", type=int, default=-1, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--task", default="train", help="train, val, test, speed or study")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--freeze", type=int, default=None, help="freeze first n model layers during training")
    parser.add_argument("--optimizer", default="SGD", help="SGD, Adam, AdamW")
    parser.add_argument("--amp", action="store_true", help="open amp")
    parser.add_argument("--project", default="tf-logs", help="save to project/name")
    parser.add_argument("--name", default="New_gate", help="save to project/name")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--resume", type=str, default="tf-logs/EMA/weights/last.pt", help="resume training from checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    task = opt.task

    data_path = resolve_path(opt.data)
    config_path = resolve_path(opt.config)
    project_path = resolve_path(opt.project)
    resume_path = resolve_path(opt.resume)
    weights_path = resolve_path(opt.weights)

    args = {
        "data": data_path,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": project_path,
        "name": opt.name,
    }
    if opt.freeze is not None:
        args["freeze"] = opt.freeze

    if task == "train":
        if resume_path and os.path.exists(resume_path):
            print(f"检测到断点，正在从 {resume_path} 恢复训练...")
            model = YOLO(resume_path)
            model.train(resume=True, **args)
        else:
            print(f"未检测到断点文件，正在根据配置 {config_path} 开启新训练...")
            model = YOLO(config_path)
            if weights_path:
                print(f"检测到预训练权重，正在从 {weights_path} 加载...")
                model.load(weights_path)
            model.train(**args)

    elif task == "val":
        model = YOLO(config_path if not weights_path else weights_path)
        model.val(**args)

    elif task == "test":
        model = YOLO(config_path if not weights_path else weights_path)
        model.val(split="test", **args)
