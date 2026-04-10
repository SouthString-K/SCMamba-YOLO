import argparse
import os
import warnings

import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

warnings.filterwarnings("ignore")


def get_weight_size(path):
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a trained SCMamba-YOLO checkpoint and export paper-ready metrics.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Dataset yaml path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--project", type=str, default="runs/val", help="Save directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or cpu")
    parser.add_argument("--save_json", action="store_true", help="Save COCO-style json if needed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = YOLO(args.weights)
    result = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        save_json=args.save_json,
    )

    if model.task == "detect":
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed["preprocess"]
        inference_time_per_image = result.speed["inference"]
        postprocess_time_per_image = result.speed["postprocess"]
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        n_l, n_p, n_g, flops = model_info(model.model)

        print("-" * 20 + " Final paper-ready statistics " + "-" * 20)

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
                f"{get_weight_size(args.weights)}MB",
            ]
        )
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrics"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx, cls_name in enumerate(model_names):
            model_metrice_table.add_row(
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
        model_metrice_table.add_row(
            [
                "all(avg)",
                f"{result.results_dict['metrics/precision(B)']:.4f}",
                f"{result.results_dict['metrics/recall(B)']:.4f}",
                f"{np.mean(result.box.f1):.4f}",
                f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                f"{np.mean(result.box.all_ap[:, 5]):.4f}",
                f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
            ]
        )
        print(model_metrice_table)

        with open(result.save_dir / "paper_data.txt", "w+", encoding="utf-8") as f:
            f.write(str(model_info_table))
            f.write("\n")
            f.write(str(model_metrice_table))

        print("-" * 20, f"Results saved to {result.save_dir}/paper_data.txt", "-" * 20)
