import csv
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from .extensions import db
from .models import VideoTask, ViolationEvent


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


def point_in_polygon(point, polygon):
    x, y = point
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def box_center_xy(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


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
                if tr.cls_name != det_cls or tid in matched_tracks:
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


def draw_zone(frame, zone, color):
    pts = np.array(zone["points"], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, 2)
    x, y = zone["points"][0]
    cv2.putText(
        frame,
        zone.get("name", "zone"),
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def process_task(app, task_id: int):
    with app.app_context():
        task = VideoTask.query.get(task_id)
        if task is None:
            return
        task.status = "processing"
        task.error_message = None
        db.session.commit()

        try:
            cfg = load_rules(app.config["RULES_FILE"])
            weights = app.config["MODEL_WEIGHTS"]
            fallback = app.config.get("FALLBACK_MODEL_WEIGHTS", "yolov8n.pt")
            model_source = weights if Path(weights).exists() else fallback
            model = YOLO(model_source)
            names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

            tracker = IoUTracker(
                match_iou=float(cfg["tracking"]["match_iou"]),
                max_lost=int(cfg["tracking"]["max_lost"]),
                stationary_iou=float(cfg["stationary"]["iou_threshold"]),
            )
            min_stationary_frames = int(cfg["stationary"]["min_frames"])

            in_path = Path(task.input_path)
            out_name = f"task_{task.id}_result.mp4"
            out_path = Path(app.config["OUTPUT_FOLDER"]) / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(in_path))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {in_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            task.total_frames = max(total, 0)
            db.session.commit()

            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

            csv_path = Path(app.config["OUTPUT_FOLDER"]) / f"task_{task.id}_events.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
                csv_writer = csv.writer(f_csv)
                csv_writer.writerow(["frame_id", "track_id", "class", "violation_type", "zone_name"])

                frame_id = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_id += 1

                    result = model.predict(frame, conf=0.35, verbose=False)[0]
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
                        current_hits = []

                        for zone in cfg["forbidden_zones"]:
                            valid_cls = zone.get("classes", [])
                            if valid_cls and tr.cls_name not in valid_cls:
                                continue
                            if point_in_polygon(center, zone["points"]):
                                current_hits.append(("no_entry", zone.get("name", "forbidden_zone")))

                        for zone in cfg["no_parking_zones"]:
                            valid_cls = zone.get("classes", [])
                            if valid_cls and tr.cls_name not in valid_cls:
                                continue
                            if tr.stationary_frames >= min_stationary_frames and point_in_polygon(center, zone["points"]):
                                current_hits.append(("illegal_parking", zone.get("name", "no_parking_zone")))

                        for vtype, zname in current_hits:
                            key = f"{vtype}:{zname}"
                            if key in tr.violations:
                                continue
                            tr.violations.add(key)
                            event = ViolationEvent(
                                task_id=task.id,
                                frame_id=frame_id,
                                track_id=tid,
                                class_name=tr.cls_name,
                                violation_type=vtype,
                                zone_name=zname,
                            )
                            db.session.add(event)
                            csv_writer.writerow([frame_id, tid, tr.cls_name, vtype, zname])

                        is_bad = len(current_hits) > 0 or len(tr.violations) > 0
                        color = (0, 0, 255) if is_bad else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID{tid} {tr.cls_name} S:{tr.stationary_frames}"
                        if tr.violations:
                            label += " VIOLATION"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

                    writer.write(frame)
                    task.processed_frames = frame_id
                    if frame_id % 30 == 0:
                        db.session.commit()

            cap.release()
            writer.release()

            task.output_path = str(out_path)
            task.status = "done"
            task.finished_at = datetime.utcnow()
            db.session.commit()

        except Exception as exc:
            task = VideoTask.query.get(task_id)
            if task:
                task.status = "failed"
                task.error_message = str(exc)
                task.finished_at = datetime.utcnow()
                db.session.commit()


def process_task_async(app, task_id: int):
    t = threading.Thread(target=process_task, args=(app, task_id), daemon=True)
    t.start()
