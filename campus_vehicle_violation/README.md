# 校园违规车辆检测系统（完整版）

本项目是一个可直接用于毕业设计演示的完整系统，包含：
- 车辆检测训练（VisDrone -> YOLO）
- 违规判定（禁行区、违停区）
- Web 前端系统（登录、上传、任务管理、统计分析）
- 单张图片检测（上传图片即返回目标框与类别统计）
- MySQL 数据库（用户、任务、违规事件）

## 1. 项目结构

```text
campus_vehicle_violation/
├─ scripts/
│  ├─ prepare_visdrone_vehicle.py
│  └─ init_db.py
├─ webapp/
│  ├─ templates/
│  ├─ static/
│  ├─ detector_service.py
│  ├─ models.py
│  ├─ auth.py
│  ├─ main.py
│  ├─ config.py
│  └─ __init__.py
├─ detect_violation.py
├─ train_yolo.py
├─ run.py
└─ requirements.txt
```

## 2. 环境准备

```bash
cd d:\bishe\campus_vehicle_violation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. MySQL 初始化

### 3.1 创建数据库

```sql
CREATE DATABASE campus_violation DEFAULT CHARACTER SET utf8mb4;
```

### 3.2 配置数据库连接

默认连接在 `webapp/config.py`：

```python
mysql+pymysql://root:123456@127.0.0.1:3306/campus_violation?charset=utf8mb4
```

可通过环境变量覆盖：

```bash
set DATABASE_URL=mysql+pymysql://root:你的密码@127.0.0.1:3306/campus_violation?charset=utf8mb4
```

### 3.3 创建表与管理员

```bash
python scripts\init_db.py --admin admin --password admin123
```

## 4. 数据预处理与训练

数据集路径（你提供）：
`D:\毕业设计\数据集\train\VisDrone2019-DET-train`

```bash
python scripts\prepare_visdrone_vehicle.py --src-root "D:\毕业设计\数据集\train\VisDrone2019-DET-train" --out-root "data\visdrone_vehicle" --val-ratio 0.2
```

```bash
python train_yolo.py --data data\visdrone_vehicle.yaml --model yolov8n.pt --epochs 80 --imgsz 960 --batch 8 --device 0
```

训练完成后模型路径通常是：
`runs\campus_vehicle_yolov8n\weights\best.pt`

## 5. 启动 Web 系统

```bash
python run.py
```

浏览器访问：
`http://127.0.0.1:5000`

功能页面：
- 登录/注册
- 仪表盘
- 新建检测任务（上传视频）
- 任务详情（状态、事件表、结果下载）
- 统计分析图（违规类型、区域、车辆类别）

## 6. 违规规则配置

编辑 `configs/campus_rules_example.yaml`，按你的摄像头画面修改多边形坐标：
- `forbidden_zones`：禁行区域
- `no_parking_zones`：禁停区域
- `stationary.min_frames`：静止判定阈值

## 7. 数据库表设计

系统自动建表：
- `users`：用户登录信息
- `video_tasks`：视频检测任务
- `violation_events`：违规事件记录

## 8. 论文可写亮点

1. 检测与规则融合：YOLO 检测 + 规则引擎判定违规行为。
2. 工程完整性：从数据处理、训练、推理到 Web 管理闭环。
3. 可视化统计：违规类型、区域热度、车辆类别分布。
4. 可扩展：后续可接 OCR 车牌识别与告警推送。
