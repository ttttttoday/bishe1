from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    tasks = db.relationship("VideoTask", backref="owner", lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class VideoTask(db.Model):
    __tablename__ = "video_tasks"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title = db.Column(db.String(120), nullable=False)
    input_filename = db.Column(db.String(255), nullable=False)
    input_path = db.Column(db.String(512), nullable=False)
    output_path = db.Column(db.String(512))
    status = db.Column(db.String(32), nullable=False, default="queued", index=True)
    total_frames = db.Column(db.Integer, default=0, nullable=False)
    processed_frames = db.Column(db.Integer, default=0, nullable=False)
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    finished_at = db.Column(db.DateTime)

    events = db.relationship("ViolationEvent", backref="task", lazy=True, cascade="all, delete-orphan")


class ViolationEvent(db.Model):
    __tablename__ = "violation_events"

    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey("video_tasks.id"), nullable=False, index=True)
    frame_id = db.Column(db.Integer, nullable=False)
    track_id = db.Column(db.Integer, nullable=False)
    class_name = db.Column(db.String(64), nullable=False)
    violation_type = db.Column(db.String(64), nullable=False, index=True)
    zone_name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
