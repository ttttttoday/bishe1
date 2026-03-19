import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def box_center_xy(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_in_polygon(point, polygon):
    x, y = point
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


@dataclass
class Track:
    tid: int
    cls_name: str
    bbox: Tuple[float, float, float, float]
    lost: int = 0
    stationary_frames: int = 0
    violations: set = field(default_factory=set)


class IoUTracker:
    def __init__(self, match_iou: float = 0.3, max_lost: int = 20, stationary_iou: float = 0.9):
        self.match_iou = match_iou
        self.max_lost = max_lost
        self.stationary_iou = stationary_iou
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: List[Tuple[Tuple[float, float, float, float], str]]):
        matched_tracks = set()
        matched_dets = set()
        existing_track_ids = list(self.tracks.keys())

        for di, (det_box, det_cls) in enumerate(detections):
            best_tid = None
            best_score = 0.0
            for tid in existing_track_ids:
                tr = self.tracks[tid]
                if tr.cls_name != det_cls:
                    continue
                if tid in matched_tracks:
                    continue
                score = iou_xyxy(det_box, tr.bbox)
                if score > best_score:
                    best_score = score
                    best_tid = tid
            if best_tid is not None and best_score >= self.match_iou:
                tr = self.tracks[best_tid]
                if iou_xyxy(tr.bbox, det_box) >= self.stationary_iou:
                    tr.stationary_frames += 1
                else:
                    tr.stationary_frames = 0
                tr.bbox = det_box
                tr.lost = 0
                matched_tracks.add(best_tid)
                matched_dets.add(di)

        for di, (det_box, det_cls) in enumerate(detections):
            if di in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid=tid, cls_name=det_cls, bbox=det_box)

        stale = []
        for tid in existing_track_ids:
            tr = self.tracks.get(tid)
            if tr is None:
                continue
            if tid not in matched_tracks:
                tr.lost += 1
            if tr.lost > self.max_lost:
                stale.append(tid)
        for tid in stale:
            self.tracks.pop(tid, None)

        return self.tracks


def load_rules(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("forbidden_zones", [])
    cfg.setdefault("no_parking_zones", [])
    cfg.setdefault("stationary", {})
    cfg.setdefault("tracking", {})
    cfg["stationary"].setdefault("min_frames", 45)
    cfg["stationary"].setdefault("iou_threshold", 0.9)
    cfg["tracking"].setdefault("match_iou", 0.3)
    cfg["tracking"].setdefault("max_lost", 20)
    return cfg


def draw_zone(frame, zone, color=(0, 180, 255)):
    pts = np.array(zone["points"], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, 2)
    name = zone.get("name", "zone")
    x, y = zone["points"][0]
    cv2.putText(frame, name, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    parser = argparse.ArgumentParser(description="Campus violation detection by zone rules.")
    parser.add_argument("--weights", required=True, help="Trained detector weight, e.g. runs/.../best.pt")
    parser.add_argument("--source", default="0", help="Video path or camera id")
    parser.add_argument("--rules", default="configs/campus_rules_example.yaml", help="Rule yaml")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--save-video", default="runs/violation_demo.mp4", help="Output annotated video")
    parser.add_argument("--save-events", default="runs/violation_events.csv", help="Output event csv")
    parser.add_argument("--show", action="store_true", help="Show window while running")
    args = parser.parse_args()

    cfg = load_rules(args.rules)
    model = YOLO(args.weights)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    tracker = IoUTracker(
        match_iou=float(cfg["tracking"]["match_iou"]),
        max_lost=int(cfg["tracking"]["max_lost"]),
        stationary_iou=float(cfg["stationary"]["iou_threshold"]),
    )

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.save_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    event_path = Path(args.save_events)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_file = open(event_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(event_file)
    csv_writer.writerow(["frame_id", "track_id", "class", "violation_type", "zone_name"])

    frame_id = 0
    min_stationary_frames = int(cfg["stationary"]["min_frames"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        result = model.predict(frame, conf=args.conf, verbose=False)[0]
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, cls_id in zip(boxes, cls_ids):
                cls_name = names.get(cls_id, str(cls_id))
                detections.append((tuple(float(x) for x in box), cls_name))

        tracks = tracker.update(detections)

        for zone in cfg["forbidden_zones"]:
            draw_zone(frame, zone, color=(0, 0, 255))
        for zone in cfg["no_parking_zones"]:
            draw_zone(frame, zone, color=(0, 180, 255))

        for tid, tr in tracks.items():
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            center = box_center_xy(tr.bbox)

            violation_hits = []
            for zone in cfg["forbidden_zones"]:
                valid_cls = zone.get("classes", [])
                if valid_cls and tr.cls_name not in valid_cls:
                    continue
                if point_in_polygon(center, zone["points"]):
                    violation_hits.append(("no_entry", zone.get("name", "forbidden_zone")))

            for zone in cfg["no_parking_zones"]:
                valid_cls = zone.get("classes", [])
                if valid_cls and tr.cls_name not in valid_cls:
                    continue
                if tr.stationary_frames >= min_stationary_frames and point_in_polygon(center, zone["points"]):
                    violation_hits.append(("illegal_parking", zone.get("name", "no_parking_zone")))

            for vtype, zname in violation_hits:
                key = f"{vtype}:{zname}"
                if key not in tr.violations:
                    tr.violations.add(key)
                    csv_writer.writerow([frame_id, tid, tr.cls_name, vtype, zname])

            is_bad = len(violation_hits) > 0 or len(tr.violations) > 0
            color = (0, 0, 255) if is_bad else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID{tid} {tr.cls_name} S:{tr.stationary_frames}"
            if tr.violations:
                label += " VIOLATION"
            cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        writer.write(frame)
        if args.show:
            cv2.imshow("Campus Violation Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    event_file.close()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Saved video: {out_path.resolve()}")
    print(f"Saved events: {event_path.resolve()}")


if __name__ == "__main__":
    main()
