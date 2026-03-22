import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "campus-violation-secret-key-change-me")
    
    # --- 修改后的部分开始 ---
    # 使用 SQLite 数据库，数据将保存在项目根目录下的 campus.db 文件中
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "sqlite:///" + str(BASE_DIR / "campus.db")
    )
    # --- 修改后的部分结束 ---

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = str(BASE_DIR / "webapp" / "uploads")
    OUTPUT_FOLDER = str(BASE_DIR / "webapp" / "outputs")
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webp"}
    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

    MODEL_WEIGHTS = os.getenv(
        "MODEL_WEIGHTS",
        str(BASE_DIR / "runs" / "campus_vehicle_yolov8n" / "weights" / "best.pt"),
    )
    FALLBACK_MODEL_WEIGHTS = os.getenv("FALLBACK_MODEL_WEIGHTS", "yolov8n.pt")
    RULES_FILE = os.getenv(
        "RULES_FILE",
        str(BASE_DIR / "configs" / "campus_rules_example.yaml"),
    )
