import argparse
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm
import yaml


VEHICLE_MAP = {
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning_tricycle",
    9: "bus",
    10: "motor",
}

VEHICLE_IDS = sorted(VEHICLE_MAP.keys())
ID2NEW = {k: i for i, k in enumerate(VEHICLE_IDS)}
NAMES = [VEHICLE_MAP[k] for k in VEHICLE_IDS]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def parse_visdrone_line(line: str):
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 8:
        return None
    x, y, w, h, score, cls_id, truncation, occlusion = parts[:8]
    return {
        "x": float(x),
        "y": float(y),
        "w": float(w),
        "h": float(h),
        "score": int(float(score)),
        "cls_id": int(float(cls_id)),
        "truncation": int(float(truncation)),
        "occlusion": int(float(occlusion)),
    }


def vis_to_yolo(img_w, img_h, x, y, w, h):
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone DET to YOLO vehicle dataset.")
    parser.add_argument("--src-root", required=True, help="Path like .../VisDrone2019-DET-train")
    parser.add_argument("--out-root", default="data/visdrone_vehicle", help="Output dataset folder")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    img_dir = src_root / "images"
    ann_dir = src_root / "annotations"

    if not img_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(f"Invalid VisDrone root: {src_root}")

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No images found in {img_dir}")

    random.seed(args.seed)
    random.shuffle(images)

    split_idx = int(len(images) * (1 - args.val_ratio))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        out_img_dir = out_root / "images" / split_name
        out_lab_dir = out_root / "labels" / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lab_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(split_imgs, desc=f"Preparing {split_name}"):
            ann_path = ann_dir / (img_path.stem + ".txt")
            if not ann_path.exists():
                continue

            import cv2
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            img_h, img_w = image.shape[:2]

            yolo_lines = []
            for line in ann_path.read_text(encoding="utf-8").splitlines():
                row = parse_visdrone_line(line)
                if row is None:
                    continue

                cls_id = row["cls_id"]
                if row["score"] != 1:
                    continue
                if cls_id not in ID2NEW:
                    continue
                if row["w"] <= 1 or row["h"] <= 1:
                    continue

                cx, cy, nw, nh = vis_to_yolo(
                    img_w, img_h, row["x"], row["y"], row["w"], row["h"]
                )
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                nw = min(max(nw, 1e-6), 1.0)
                nh = min(max(nh, 1e-6), 1.0)
                yolo_cls = ID2NEW[cls_id]
                yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            link_or_copy(img_path, out_img_dir / img_path.name)
            (out_lab_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(yolo_lines), encoding="utf-8"
            )

    dataset_yaml = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(NAMES)},
        "nc": len(NAMES),
    }

    yaml_path = out_root.parent / "visdrone_vehicle.yaml"
    yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(f"Done. Dataset yaml: {yaml_path.resolve()}")
    print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")
    print(f"Classes: {NAMES}")


if __name__ == "__main__":
    main()
