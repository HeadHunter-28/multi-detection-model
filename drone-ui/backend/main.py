"""
Drone Detection API — FastAPI Backend
Exposes two inference endpoints:
  POST /api/benchmark   → Run single image through all 4 models (YOLOv8s, RT-DETR-L, Faster-RCNN, SSD)
  POST /api/ensemble    → Run WBF ensemble + Cascade, return both outputs
"""

import os
import sys
import time
import io
import base64
import warnings
import traceback
from typing import Optional

import numpy as np
import cv2
import torch
import torchvision
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "Apple MPS"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = f"CUDA: {torch.cuda.get_device_name(0)}"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

UL_DEVICE = "mps" if DEVICE.type == "mps" else (0 if DEVICE.type == "cuda" else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
YOLO_WEIGHTS  = os.path.join(BASE_DIR, "drone_yolov8s_final.pt")
DETR_WEIGHTS  = os.path.join(BASE_DIR, "best_detr.pt")
FRCNN_WEIGHTS = os.path.join(BASE_DIR, "fasterrcnn_drone.pth")
SSD_WEIGHTS   = os.path.join(BASE_DIR, "ssd_drone_model_kaggle.pth")

# YOLO_WEIGHTS  = os.path.abspath('./drone_yolov8s_final.pt')
# DETR_WEIGHTS  = os.path.abspath('./best_detr.pt')
# FRCNN_WEIGHTS = os.path.abspath('./fasterrcnn_drone.pth')
# SSD_WEIGHTS   = os.path.abspath('./ssd_drone_model_kaggle.pth')

CLASS_NAMES = ["AirPlane", "Drone", "Helicopter"]
NUM_CLASSES = len(CLASS_NAMES)
CONF_THRESH = 0.30
IMG_SIZE    = 640

# ── Model builders ────────────────────────────────────────────────────────────
def build_fasterrcnn(weights_path=None):
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES + 1)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    return model.to(DEVICE).eval()

def build_ssd(weights_path=None):
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.ssd import SSDClassificationHead
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    in_ch = [672, 480, 512, 256, 256, 128]
    n_anc = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_ch, n_anc, NUM_CLASSES + 1)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    return model.to(DEVICE).eval()

# ── Lazy model loading ────────────────────────────────────────────────────────
_models = {}

def get_models():
    global _models
    if _models:
        return _models
    print("Loading all models…")
    try:
        from ultralytics import YOLO, RTDETR
        if os.path.exists(YOLO_WEIGHTS):
            _models["YOLOv8s"] = YOLO(YOLO_WEIGHTS)
            print("  ✅ YOLOv8s loaded")
        if os.path.exists(DETR_WEIGHTS):
            _models["RT-DETR-L"] = RTDETR(DETR_WEIGHTS)
            print("  ✅ RT-DETR-L loaded")
    except Exception as e:
        print(f"  ⚠️  Ultralytics models failed: {e}")

    try:
        _models["Faster-RCNN"] = build_fasterrcnn(FRCNN_WEIGHTS)
        print("  ✅ Faster-RCNN loaded")
    except Exception as e:
        print(f"  ⚠️  Faster-RCNN failed: {e}")

    try:
        _models["SSD"] = build_ssd(SSD_WEIGHTS)
        print("  ✅ SSD loaded")
    except Exception as e:
        print(f"  ⚠️  SSD failed: {e}")

    print(f"  🎯 {len(_models)}/4 models loaded on {DEVICE_NAME}")
    return _models

# ── Inference helpers ─────────────────────────────────────────────────────────
def infer_ultralytics(model, img_np: np.ndarray, conf: float = CONF_THRESH):
    t0 = time.perf_counter()
    res = model.predict(img_np, imgsz=IMG_SIZE, conf=conf, device=UL_DEVICE, verbose=False)
    ms = (time.perf_counter() - t0) * 1000.0
    r = res[0]
    if r.boxes and len(r.boxes):
        return (
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            ms,
        )
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), ms

def infer_torchvision(model, img_np: np.ndarray, conf: float = CONF_THRESH, sz=None):
    h, w = img_np.shape[:2]
    inp = img_np if sz is None else cv2.resize(img_np, (sz, sz))
    ih, iw = inp.shape[:2]
    inp_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(inp_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(t)[0]
    ms = (time.perf_counter() - t0) * 1000.0
    mask = pred["scores"].cpu().numpy() >= conf
    boxes  = pred["boxes"].cpu().numpy()[mask]
    scores = pred["scores"].cpu().numpy()[mask]
    labels = (pred["labels"].cpu().numpy()[mask] - 1).clip(min=0)
    if sz and len(boxes):
        boxes[:, 0] *= w / iw; boxes[:, 2] *= w / iw
        boxes[:, 1] *= h / ih; boxes[:, 3] *= h / ih
    return boxes, scores, labels, ms

def boxes_to_list(boxes, scores, labels):
    results = []
    for box, score, lbl in zip(boxes, scores, labels):
        cls = int(lbl)
        name = CLASS_NAMES[cls] if 0 <= cls < NUM_CLASSES else "Unknown"
        results.append({
            "x1": float(box[0]), "y1": float(box[1]),
            "x2": float(box[2]), "y2": float(box[3]),
            "score": float(score),
            "class_id": cls,
            "class_name": name,
        })
    return results

# ── WBF ───────────────────────────────────────────────────────────────────────
def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def wbf(all_boxes, all_scores, all_labels, iou_thr=0.50, skip_box_thr=0.01):
    boxes  = np.concatenate(all_boxes,  axis=0) if any(len(b) > 0 for b in all_boxes)  else np.zeros((0, 4))
    scores = np.concatenate(all_scores, axis=0) if any(len(s) > 0 for s in all_scores) else np.zeros(0)
    labels = np.concatenate(all_labels, axis=0) if any(len(l) > 0 for l in all_labels) else np.zeros(0, int)

    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)

    keep = scores >= skip_box_thr
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    order = np.argsort(-scores)
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    clusters_b, clusters_s, clusters_l = [], [], []
    for i in range(len(boxes)):
        matched = -1
        for j, (cb, cs_, cl) in enumerate(zip(clusters_b, clusters_s, clusters_l)):
            rep_box = np.average(cb, axis=0, weights=cs_)
            if cl[0] == labels[i] and iou(rep_box, boxes[i]) >= iou_thr:
                matched = j
                break
        if matched >= 0:
            clusters_b[matched].append(boxes[i])
            clusters_s[matched].append(scores[i])
            clusters_l[matched].append(labels[i])
        else:
            clusters_b.append([boxes[i]])
            clusters_s.append([scores[i]])
            clusters_l.append([labels[i]])

    out_boxes, out_scores, out_labels = [], [], []
    for cb, cs_, cl in zip(clusters_b, clusters_s, clusters_l):
        w = np.array(cs_)
        out_boxes.append(np.average(cb, axis=0, weights=w))
        out_scores.append(float(np.mean(w)))
        out_labels.append(int(np.bincount(cl).argmax()))

    return np.array(out_boxes), np.array(out_scores), np.array(out_labels, int)

# ── Image helpers ─────────────────────────────────────────────────────────────
def decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def apply_auto_padding(img: np.ndarray, scale_factor=0.15) -> np.ndarray:
    """Pad the image with its edge colors so it appears scaled down recursively"""
    h, w = img.shape[:2]
    new_h, new_w = int(h / scale_factor), int(w / scale_factor)
    
    pad_h = (new_h - h) // 2
    pad_w = (new_w - w) // 2
    
    # Calculate median color from edges
    edges = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]], axis=0)
    bg_color = np.median(edges, axis=0).astype(np.uint8).tolist()
    
    padded = cv2.copyMakeBorder(
        img, 
        pad_h, new_h - h - pad_h,
        pad_w, new_w - w - pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=bg_color
    )
    return padded

def encode_image_b64(img_np: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_np)
    return base64.b64encode(buf).decode("utf-8")

def draw_boxes_on_img(img: np.ndarray, boxes, scores, labels, color=(52, 211, 153), label_prefix="") -> np.ndarray:
    vis = img.copy()
    bgr_color = (color[2], color[1], color[0])
    for box, score, lbl in zip(boxes, scores, labels):
        cls = int(lbl)
        name = CLASS_NAMES[cls] if 0 <= cls < NUM_CLASSES else "?"
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), bgr_color, 2)
        cv2.putText(vis, f"{label_prefix}{name} {score:.2f}", (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr_color, 2)
    return vis

MODEL_COLORS = {
    "YOLOv8s":    (52, 211, 153),
    "RT-DETR-L":  (251, 146, 60),
    "Faster-RCNN":(99, 102, 241),
    "SSD":        (236, 72, 153),
    "Ensemble":   (255, 215, 0),
    "Cascade":    (0, 200, 255),
}

MODEL_ORDER = ["YOLOv8s", "RT-DETR-L", "Faster-RCNN", "SSD"]
MODEL_ALIASES = {
    "yolo": "YOLOv8s",
    "yolov8s": "YOLOv8s",
    "detr": "RT-DETR-L",
    "rtdetr": "RT-DETR-L",
    "rt-detr-l": "RT-DETR-L",
    "frcnn": "Faster-RCNN",
    "fasterrcnn": "Faster-RCNN",
    "faster-rcnn": "Faster-RCNN",
    "ssd": "SSD",
}


def parse_requested_models(models_raw: Optional[str], available_models: dict):
    requested = (models_raw or "all").strip().lower()
    if requested == "all":
        selected = [m for m in MODEL_ORDER if m in available_models]
        include_ensemble = True
        return selected, include_ensemble

    include_ensemble = False
    selected = []
    for token in [x.strip().lower() for x in requested.split(",") if x.strip()]:
        if token == "ensemble":
            include_ensemble = True
            continue
        mapped = MODEL_ALIASES.get(token)
        if mapped and mapped in available_models and mapped not in selected:
            selected.append(mapped)

    return selected, include_ensemble


def detect_with_model(model_name: str, model_obj, img_np: np.ndarray, conf: float):
    preprocess_start = time.perf_counter()
    if model_name in ("YOLOv8s", "RT-DETR-L"):
        boxes, scores, labels, infer_ms = infer_ultralytics(model_obj, img_np, conf=conf)
    elif model_name == "Faster-RCNN":
        boxes, scores, labels, infer_ms = infer_torchvision(model_obj, img_np, conf=conf, sz=None)
    else:
        boxes, scores, labels, infer_ms = infer_torchvision(model_obj, img_np, conf=conf, sz=320)

    total_ms = (time.perf_counter() - preprocess_start) * 1000.0
    postprocess_ms = max(0.0, total_ms - infer_ms)

    detections = []
    for box, score, lbl in zip(boxes, scores, labels):
        cls_id = int(lbl)
        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < NUM_CLASSES else "Unknown"
        detections.append({
            "class": cls_name,
            "class_id": cls_id,
            "confidence": float(score),
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
        })

    metrics = {
        "preprocess_ms": round(max(0.0, total_ms - infer_ms - postprocess_ms), 2),
        "inference_ms": round(float(infer_ms), 2),
        "postprocess_ms": round(float(postprocess_ms), 2),
        "total_ms": round(float(total_ms), 2),
        "fps": round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0,
        "objects_found": len(detections),
        "avg_confidence": round(float(np.mean(scores)), 3) if len(scores) else 0.0,
        "max_confidence": round(float(np.max(scores)), 3) if len(scores) else 0.0,
    }

    return boxes, scores, labels, detections, metrics


def build_comparison(per_model_benchmark: dict, detections_by_model: dict):
    if not per_model_benchmark:
        return {
            "fastest": None,
            "most_accurate": None,
            "best_overall": None,
            "most_detections": None,
        }

    valid_models = list(per_model_benchmark.keys())
    fastest = min(valid_models, key=lambda m: per_model_benchmark[m].get("total_ms", float("inf")))
    most_accurate = max(valid_models, key=lambda m: per_model_benchmark[m].get("avg_confidence", 0.0))
    most_detections = max(valid_models, key=lambda m: len(detections_by_model.get(m, [])))

    # simple blended score for dashboard summary
    def score(m):
        b = per_model_benchmark[m]
        speed = 1.0 / max(b.get("total_ms", 1.0), 1.0)
        conf = b.get("avg_confidence", 0.0)
        count = b.get("objects_found", 0)
        return (0.45 * speed) + (0.4 * conf) + (0.15 * min(count / 10.0, 1.0))

    best_overall = max(valid_models, key=score)
    return {
        "fastest": fastest,
        "most_accurate": most_accurate,
        "best_overall": best_overall,
        "most_detections": most_detections,
    }


MODEL_KEY_MAP = {
    "YOLOv8s": "yolo",
    "RT-DETR-L": "detr",
    "Faster-RCNN": "frcnn",
    "SSD": "ssd",
}


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Drone Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    get_models()

@app.get("/api/health")
async def health():
    models = get_models()
    return {
        "status": "ok",
        "device": DEVICE_NAME,
        "models_loaded": list(models.keys()),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
    }

@app.post("/api/benchmark")
async def benchmark_endpoint(
    image: UploadFile = File(...),
    models: Optional[str] = Form("all"),
    conf_threshold: float = Form(0.30),
    pad_image: bool = Form(False),
    pad_scale: float = Form(0.15),
):
    """
    Contract-aligned endpoint:
      POST /api/benchmark
      - image: uploaded file
      - models: comma-separated (e.g. "yolo,ensemble") or "all"
      - conf_threshold: float (default 0.30)
    """
    request_start = time.perf_counter()
    try:
        data = await image.read()
        img = decode_image(data)
        if pad_image:
            img = apply_auto_padding(img, scale_factor=pad_scale)

        h, w = img.shape[:2]
        loaded_models = get_models()
        selected_models, include_ensemble = parse_requested_models(models, loaded_models)

        if not selected_models and not include_ensemble:
            raise HTTPException(
                status_code=400,
                detail="No valid models requested. Use models=all or comma-separated: yolo,detr,frcnn,ssd,ensemble",
            )

        detections_by_model = {}
        benchmark_by_model = {}
        all_boxes, all_scores, all_labels = [], [], []

        for mname in selected_models:
            model_obj = loaded_models.get(mname)
            if model_obj is None:
                continue

            boxes, scores, labels, detections, metrics = detect_with_model(
                mname, model_obj, img, conf=conf_threshold
            )
            key = MODEL_KEY_MAP[mname]
            detections_by_model[key] = detections
            benchmark_by_model[key] = metrics
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if include_ensemble and all_boxes:
            t0 = time.perf_counter()
            fused_b, fused_s, fused_l = wbf(all_boxes, all_scores, all_labels)
            fusion_ms = (time.perf_counter() - t0) * 1000.0

            ensemble_detections = []
            for box, score, lbl in zip(fused_b, fused_s, fused_l):
                cls_id = int(lbl)
                cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < NUM_CLASSES else "Unknown"
                ensemble_detections.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": float(score),
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                })

            detections_by_model["ensemble"] = ensemble_detections
            benchmark_by_model["ensemble"] = {
                "fusion_ms": round(float(fusion_ms), 2),
                "total_ms": round(float(fusion_ms), 2),
                "objects_found": len(ensemble_detections),
                "avg_confidence": round(float(np.mean(fused_s)), 3) if len(fused_s) else 0.0,
                "max_confidence": round(float(np.max(fused_s)), 3) if len(fused_s) else 0.0,
            }

        total_request_ms = (time.perf_counter() - request_start) * 1000.0

        response = {
            "image": {
                "width": int(w),
                "height": int(h),
                "annotated_base64": f"data:image/jpeg;base64,{encode_image_b64(img)}",
            },
            "detections": detections_by_model,
            "benchmark": {
                "per_model": benchmark_by_model,
                "ensemble": benchmark_by_model.get("ensemble"),
                "total_request_ms": round(float(total_request_ms), 2),
            },
            "comparison": build_comparison(
                {k: v for k, v in benchmark_by_model.items() if k != "ensemble"},
                detections_by_model,
            ),
        }
        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ensemble")
async def ensemble_endpoint(file: UploadFile = File(...), pad_image: bool = Form(False), pad_scale: float = Form(0.15), conf_thresh: float = Form(0.30)):
    """
    Run an uploaded image through WBF ensemble + Speed-Accuracy Cascade.
    Returns both outputs with metrics and annotated images.
    """
    try:
        data = await file.read()
        img = decode_image(data)
        if pad_image:
            img = apply_auto_padding(img, scale_factor=pad_scale)
        models = get_models()

        # ── WBF Ensemble ───────────────────────────────────────────────────────
        all_b, all_s, all_l = [], [], []
        individual_results = {}
        t0_wbf = time.perf_counter()
        MODEL_ORDER = ["YOLOv8s", "RT-DETR-L", "Faster-RCNN", "SSD"]

        for mname in MODEL_ORDER:
            model = models.get(mname)
            if model is None:
                all_b.append(np.zeros((0, 4)))
                all_s.append(np.zeros(0))
                all_l.append(np.zeros(0, int))
                continue
            if mname in ("YOLOv8s", "RT-DETR-L"):
                b, s, l, ms = infer_ultralytics(model, img, conf=conf_thresh)
            elif mname == "Faster-RCNN":
                b, s, l, ms = infer_torchvision(model, img, conf=conf_thresh, sz=None)
            else:
                b, s, l, ms = infer_torchvision(model, img, conf=conf_thresh, sz=320)
            all_b.append(b)
            all_s.append(s)
            all_l.append(l)
            individual_results[mname] = {
                "detections": boxes_to_list(b, s, l),
                "latency_ms": round(float(ms), 2),
                "count": int(len(s)),
            }

        wbf_total_ms = (time.perf_counter() - t0_wbf) * 1000
        fused_b, fused_s, fused_l = wbf(all_b, all_s, all_l)

        wbf_vis = draw_boxes_on_img(img, fused_b, fused_s, fused_l,
                                    color=MODEL_COLORS["Ensemble"])
        wbf_img_b64 = encode_image_b64(wbf_vis)

        # ── Cascade ────────────────────────────────────────────────────────────
        fast_conf = 0.70
        preds_casc = {}
        stages_used = []
        t0_casc = time.perf_counter()

        # Stage 1: SSD
        if "SSD" in models:
            b, s, l, _ = infer_torchvision(models["SSD"], img, conf=conf_thresh, sz=320)
            preds_casc["SSD"] = (b, s, l)
            stages_used.append("SSD")
            if len(s) and s.max() >= fast_conf:
                casc_ms = (time.perf_counter() - t0_casc) * 1000
                casc_b, casc_s, casc_l = b, s, l
                casc_stopped = "Stage 1 (SSD only)"
            else:
                # Stage 2: YOLOv8s
                if "YOLOv8s" in models:
                    b2, s2, l2, _ = infer_ultralytics(models["YOLOv8s"], img, conf=conf_thresh)
                    preds_casc["YOLOv8s"] = (b2, s2, l2)
                    stages_used.append("YOLOv8s")
                    all_b2 = [v[0] for v in preds_casc.values()]
                    all_s2 = [v[1] for v in preds_casc.values()]
                    all_l2 = [v[2] for v in preds_casc.values()]
                    fb2, fs2, fl2 = wbf(all_b2, all_s2, all_l2)
                    if len(fs2) and fs2.max() >= fast_conf:
                        casc_ms = (time.perf_counter() - t0_casc) * 1000
                        casc_b, casc_s, casc_l = fb2, fs2, fl2
                        casc_stopped = "Stage 2 (SSD + YOLOv8s)"
                    else:
                        # Stage 3: RT-DETR
                        if "RT-DETR-L" in models:
                            b3, s3, l3, _ = infer_ultralytics(models["RT-DETR-L"], img, conf=conf_thresh)
                            preds_casc["RT-DETR-L"] = (b3, s3, l3)
                            stages_used.append("RT-DETR-L")
                        all_b3 = [v[0] for v in preds_casc.values()]
                        all_s3 = [v[1] for v in preds_casc.values()]
                        all_l3 = [v[2] for v in preds_casc.values()]
                        casc_b, casc_s, casc_l = wbf(all_b3, all_s3, all_l3)
                        casc_ms = (time.perf_counter() - t0_casc) * 1000
                        casc_stopped = "Stage 3 (all models)"
                else:
                    casc_ms = (time.perf_counter() - t0_casc) * 1000
                    casc_b, casc_s, casc_l = b, s, l
                    casc_stopped = "Stage 1 (SSD fallback)"
        else:
            # Fallback: just WBF
            casc_b, casc_s, casc_l = fused_b, fused_s, fused_l
            casc_ms = wbf_total_ms
            casc_stopped = "WBF (SSD not loaded)"
            stages_used = list(individual_results.keys())

        casc_vis = draw_boxes_on_img(img, casc_b, casc_s, casc_l,
                                     color=MODEL_COLORS["Cascade"])
        casc_img_b64 = encode_image_b64(casc_vis)

        # Original image (no annotations)
        orig_b64 = encode_image_b64(img)

        return JSONResponse({
            "individual": individual_results,
            "wbf": {
                "detections": boxes_to_list(fused_b, fused_s, fused_l),
                "total_ms": round(float(wbf_total_ms), 2),
                "count": int(len(fused_s)),
                "annotated_image": wbf_img_b64,
            },
            "cascade": {
                "detections": boxes_to_list(casc_b, casc_s, casc_l),
                "total_ms": round(float(casc_ms), 2),
                "count": int(len(casc_s)),
                "stages_used": stages_used,
                "stopped_at": casc_stopped,
                "annotated_image": casc_img_b64,
            },
            "original_image": orig_b64,
            "device": DEVICE_NAME,
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
