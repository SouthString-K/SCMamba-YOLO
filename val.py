from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

from ultralytics import YOLO
from ultralytics.nn.modules.enhance_front import configure_enhance_runtime
from ultralytics.utils.torch_utils import model_info


def resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def get_weight_size(path: str) -> str:
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="detector checkpoint (.pt)")
    parser.add_argument("--data", type=str, required=True, help="dataset yaml path")
    parser.add_argument("--split", type=str, default="test", help="val or test")
    parser.add_argument("--imgsz", type=int, default=640, help="input size")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--device", default="0", help="cuda device or cpu")
    parser.add_argument("--project", default="runs/val", help="save to project/name")
    parser.add_argument("--name", default="scmambayolo_val", help="save to project/name")
    parser.add_argument("--save_json", action="store_true", help="export json for COCO-style eval")
    parser.add_argument("--enhance_weights", type=str, default=None, help="optional InteractNet checkpoint (.pth)")
    parser.add_argument(
        "--enhance_freeze",
        action="store_true",
        help="kept for symmetry with train.py; inference always uses eval mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    weights_path = resolve_path(opt.weights)

    configure_enhance_runtime(
        enable=True,
        weights=resolve_path(opt.enhance_weights),
        freeze=opt.enhance_freeze,
    )

    model = YOLO(weights_path)
    result = model.val(
        data=resolve_path(opt.data),
        split=opt.split,
        imgsz=opt.imgsz,
        batch=opt.batch,
        project=resolve_path(opt.project),
        name=opt.name,
        device=opt.device,
        save_json=opt.save_json,
    )

    if model.task == "detect":
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed["preprocess"]
        inference_time_per_image = result.speed["inference"]
        postprocess_time_per_image = result.speed["postprocess"]
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        _, n_p, _, flops = model_info(model.model)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = [
            "GFLOPs",
            "Parameters",
            "Preprocess / img",
            "Inference / img",
            "Postprocess / img",
            "FPS(total)",
            "FPS(inference)",
            "Model File Size",
        ]
        model_info_table.add_row(
            [
                f"{flops:.1f}",
                f"{n_p:,}",
                f"{preprocess_time_per_image / 1000:.6f}s",
                f"{inference_time_per_image / 1000:.6f}s",
                f"{postprocess_time_per_image / 1000:.6f}s",
                f"{1000 / all_time_per_image:.2f}",
                f"{1000 / inference_time_per_image:.2f}",
                f"{get_weight_size(weights_path)}MB",
            ]
        )
        print(model_info_table)

        model_metric_table = PrettyTable()
        model_metric_table.title = "Model Metrics"
        model_metric_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx, cls_name in enumerate(model_names):
            model_metric_table.add_row(
                [
                    cls_name,
                    f"{result.box.p[idx]:.4f}",
                    f"{result.box.r[idx]:.4f}",
                    f"{result.box.f1[idx]:.4f}",
                    f"{result.box.ap50[idx]:.4f}",
                    f"{result.box.all_ap[idx, 5]:.4f}",
                    f"{result.box.ap[idx]:.4f}",
                ]
            )

        model_metric_table.add_row(
            [
                "all(mean)",
                f"{result.results_dict['metrics/precision(B)']:.4f}",
                f"{result.results_dict['metrics/recall(B)']:.4f}",
                f"{np.mean(result.box.f1):.4f}",
                f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                f"{np.mean(result.box.all_ap[:, 5]):.4f}",
                f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
            ]
        )
        print(model_metric_table)

        save_path = Path(result.save_dir) / "paper_data.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(str(model_info_table))
            f.write("\n")
            f.write(str(model_metric_table))

        print(f"Saved summary to {save_path}")
