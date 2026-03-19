import os
import uuid
from collections import Counter
from pathlib import Path

import cv2
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import func
from ultralytics import YOLO

from .detector_service import process_task_async
from .extensions import db
from .models import VideoTask, ViolationEvent


main_bp = Blueprint("main", __name__)


def allowed_file(filename, allowed_ext):
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in allowed_ext


@main_bp.route("/")
@login_required
def dashboard():
    total_tasks = VideoTask.query.filter_by(user_id=current_user.id).count()
    done_tasks = VideoTask.query.filter_by(user_id=current_user.id, status="done").count()
    failed_tasks = VideoTask.query.filter_by(user_id=current_user.id, status="failed").count()

    event_count = (
        db.session.query(func.count(ViolationEvent.id))
        .join(VideoTask, ViolationEvent.task_id == VideoTask.id)
        .filter(VideoTask.user_id == current_user.id)
        .scalar()
        or 0
    )
    latest_tasks = (
        VideoTask.query.filter_by(user_id=current_user.id)
        .order_by(VideoTask.created_at.desc())
        .limit(8)
        .all()
    )
    return render_template(
        "dashboard.html",
        total_tasks=total_tasks,
        done_tasks=done_tasks,
        failed_tasks=failed_tasks,
        event_count=event_count,
        latest_tasks=latest_tasks,
    )


@main_bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        title = request.form.get("title", "").strip() or "untitled_task"
        file = request.files.get("video")
        if file is None or file.filename == "":
            flash("请选择视频文件。", "warning")
            return redirect(url_for("main.upload"))
        if not allowed_file(file.filename, current_app.config["ALLOWED_EXTENSIONS"]):
            flash("视频格式不支持，请上传 mp4/avi/mov/mkv。", "warning")
            return redirect(url_for("main.upload"))

        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / unique_name
        file.save(save_path)

        task = VideoTask(
            user_id=current_user.id,
            title=title,
            input_filename=file.filename,
            input_path=str(save_path),
            status="queued",
        )
        db.session.add(task)
        db.session.commit()

        process_task_async(current_app._get_current_object(), task.id)
        flash("任务已创建，正在后台处理。", "success")
        return redirect(url_for("main.task_detail", task_id=task.id))

    return render_template("upload.html")


@main_bp.route("/image-detect", methods=["GET", "POST"])
@login_required
def image_detect():
    if request.method == "POST":
        image = request.files.get("image")
        if image is None or image.filename == "":
            flash("请选择图片文件。", "warning")
            return redirect(url_for("main.image_detect"))
        if not allowed_file(image.filename, current_app.config["IMAGE_EXTENSIONS"]):
            flash("图片格式不支持，请上传 jpg/jpeg/png/bmp/webp。", "warning")
            return redirect(url_for("main.image_detect"))

        ext = image.filename.rsplit(".", 1)[1].lower()
        image_name = f"img_{uuid.uuid4().hex}.{ext}"
        in_path = Path(current_app.config["UPLOAD_FOLDER"]) / image_name
        in_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(in_path)

        weights = current_app.config["MODEL_WEIGHTS"]
        fallback = current_app.config["FALLBACK_MODEL_WEIGHTS"]
        model_source = weights if Path(weights).exists() else fallback
        if model_source == fallback:
            flash(f"未找到训练权重，已自动使用初始模型：{fallback}", "info")

        model = YOLO(model_source)
        result = model.predict(str(in_path), conf=0.35, verbose=False)[0]
        plotted = result.plot()

        out_name = f"img_result_{uuid.uuid4().hex}.jpg"
        out_path = Path(current_app.config["OUTPUT_FOLDER"]) / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), plotted)

        cls_ids = []
        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()

        names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
        cls_counter = Counter([names.get(i, str(i)) for i in cls_ids])
        stats = sorted(cls_counter.items(), key=lambda x: x[1], reverse=True)

        return render_template(
            "image_detect.html",
            result_name=out_name,
            total=len(cls_ids),
            stats=stats,
        )

    return render_template("image_detect.html", result_name=None, total=0, stats=[])


@main_bp.route("/outputs/<path:filename>")
@login_required
def output_file(filename):
    path = Path(current_app.config["OUTPUT_FOLDER"]) / filename
    if not path.exists():
        flash("文件不存在。", "warning")
        return redirect(url_for("main.dashboard"))
    return send_file(str(path))


@main_bp.route("/tasks")
@login_required
def tasks():
    rows = VideoTask.query.filter_by(user_id=current_user.id).order_by(VideoTask.created_at.desc()).all()
    return render_template("tasks.html", tasks=rows)


@main_bp.route("/tasks/<int:task_id>")
@login_required
def task_detail(task_id):
    task = VideoTask.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    events = ViolationEvent.query.filter_by(task_id=task.id).order_by(ViolationEvent.frame_id.asc()).limit(500).all()
    return render_template("task_detail.html", task=task, events=events)


@main_bp.route("/stats")
@login_required
def stats():
    type_rows = (
        db.session.query(ViolationEvent.violation_type, func.count(ViolationEvent.id))
        .join(VideoTask, ViolationEvent.task_id == VideoTask.id)
        .filter(VideoTask.user_id == current_user.id)
        .group_by(ViolationEvent.violation_type)
        .all()
    )
    zone_rows = (
        db.session.query(ViolationEvent.zone_name, func.count(ViolationEvent.id))
        .join(VideoTask, ViolationEvent.task_id == VideoTask.id)
        .filter(VideoTask.user_id == current_user.id)
        .group_by(ViolationEvent.zone_name)
        .order_by(func.count(ViolationEvent.id).desc())
        .limit(10)
        .all()
    )
    class_rows = (
        db.session.query(ViolationEvent.class_name, func.count(ViolationEvent.id))
        .join(VideoTask, ViolationEvent.task_id == VideoTask.id)
        .filter(VideoTask.user_id == current_user.id)
        .group_by(ViolationEvent.class_name)
        .all()
    )
    return render_template("stats.html", type_rows=type_rows, zone_rows=zone_rows, class_rows=class_rows)


@main_bp.route("/download/video/<int:task_id>")
@login_required
def download_video(task_id):
    task = VideoTask.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    if not task.output_path or not os.path.exists(task.output_path):
        flash("结果视频尚未生成。", "warning")
        return redirect(url_for("main.task_detail", task_id=task.id))
    return send_file(task.output_path, as_attachment=True)


@main_bp.route("/download/events/<int:task_id>")
@login_required
def download_events(task_id):
    task = VideoTask.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    csv_path = Path(current_app.config["OUTPUT_FOLDER"]) / f"task_{task.id}_events.csv"
    if not csv_path.exists():
        flash("事件 CSV 尚未生成。", "warning")
        return redirect(url_for("main.task_detail", task_id=task.id))
    return send_file(str(csv_path), as_attachment=True)
