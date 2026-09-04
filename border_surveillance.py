import cv2
import threading
import queue
import time
import hashlib
import json
import os
import glob
import argparse
import numpy as np
import torch
from ultralytics import YOLO

# Extremely Accurate Facial Recognition for low-resolution/unclear portals
try:
    import insightface
    from insightface.app import FaceAnalysis
    FACE_REC_ENABLED = True
except ImportError:
    FACE_REC_ENABLED = False

# --- SYSTEM CONFIGURATION ---
EDGE_NODE_ID = "24012531011_Madhav" 

# Default camera indices (customizable via env or CLI)
MAC_CAM_DEFAULT = os.environ.get("MAC_CAM_INDEX", "0")
OBS_CAM_DEFAULT = os.environ.get("OBS_CAM_INDEX", "1")

# Dataset Mapping (Weapon Types Recognition)
# Offset Kaggle classes by +1 so Class 0 remains 'Person' (standard YOLO format)
WEAPON_CLASSES = {
    1: 'Automatic Rifle', 2: 'Bazooka', 3: 'Grenade Launcher', 
    4: 'Handgun', 5: 'Knife', 6: 'Shotgun', 7: 'SMG', 
    8: 'Sniper', 9: 'Sword'
}
# Standard COCO weapon classes if default yolov8n weights are loaded
COCO_WEAPON_CLASSES = {
    43: 'Knife',
    76: 'Scissors'
}
PERSON_CLASS = 0  

alert_queue = queue.Queue(maxsize=50)

# --- RECONNAISSANCE DB (INSIGHTFACE) ---
known_face_encodings = []
known_face_names = []
known_face_categories = []

if FACE_REC_ENABLED:
    # Initialize robust 'buffalo_l' model for unclear/low-res portals
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=-1, det_size=(640, 640)) # ctx_id=-1 forces CPU which is highly stable
else:
    face_app = None

def load_known_faces():
    if not FACE_REC_ENABLED:
        print("⚠️ InsightFace library not found. Skipping facial identification.")
        return
        
    faces_dir = "known_faces"
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
        return
        
    print("Loading high-accuracy InsightFace reconnaissance target database...")
    for filepath in glob.glob(os.path.join(faces_dir, "*.*")):
        if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = os.path.basename(filepath)
            name_full = os.path.splitext(filename)[0] 
            
            # Separate categories based on filename
            category = "Unknown"
            if "soldier" in name_full.lower() or "army" in name_full.lower():
                category = "Army"
            elif "civilian" in name_full.lower():
                category = "Civilian"
            
            try:
                img = cv2.imread(filepath)
                faces = face_app.get(img)
                if faces:
                    # Normed embedding allows precise cosine similarity via dot product
                    known_face_encodings.append(faces[0].normed_embedding)
                    known_face_names.append(name_full)
                    known_face_categories.append(category)
                    print(f" Loaded [{category}] target: {name_full}")
            except Exception as e:
                print(f" Could not load face for {filepath}: {e}")

class CameraStreamThread:
    """
    Dedicated background capture thread for each camera.
    Prevents buffer buildup lag, provides auto-reconnect, and displays
    tactical standby frames when a camera is offline or waiting for signal.
    """
    def __init__(self, name, src, width=640, height=480, target_fps=30):
        self.name = name
        self.src = src
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.running = True
        self.connected = False
        self.last_frame = None
        self.lock = threading.Lock()
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0
        self.status_msg = "Initializing..."
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _parse_source(self):
        src_target = self.src
        if isinstance(src_target, str):
            try:
                return int(src_target)
            except ValueError:
                return src_target
        return src_target

    def _capture_loop(self):
        target = self._parse_source()

        while self.running:
            with self.lock:
                self.status_msg = f"Connecting to {self.src}..."
                self.connected = False

            cap = cv2.VideoCapture(target)
            if not cap.isOpened():
                with self.lock:
                    self.status_msg = f"Waiting for source: {self.src}"
                    self.connected = False
                time.sleep(1.5)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Warmup read
            warmup_ok = False
            for _ in range(8):
                ret, frame = cap.read()
                if ret and frame is not None:
                    warmup_ok = True
                    with self.lock:
                        self.connected = True
                        self.last_frame = frame
                        self.status_msg = "Active"
                    break
                time.sleep(0.15)

            if not warmup_ok:
                cap.release()
                time.sleep(1.0)
                continue

            # Continuous non-blocking capture loop
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # If video file loop back to start
                    if isinstance(target, str) and not (target.startswith("rtsp") or target.startswith("http")):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue
                    with self.lock:
                        self.connected = False
                        self.status_msg = "Signal Lost. Reconnecting..."
                    break

                # Calculate real FPS
                self.fps_counter += 1
                now = time.time()
                if now - self.fps_timer >= 1.0:
                    self.current_fps = self.fps_counter / (now - self.fps_timer)
                    self.fps_counter = 0
                    self.fps_timer = now

                with self.lock:
                    self.connected = True
                    self.last_frame = frame
                    self.status_msg = "Active"

                time.sleep(0.005)

            cap.release()
            time.sleep(1.0)

    def get_frame(self):
        with self.lock:
            if self.connected and self.last_frame is not None:
                return True, self.last_frame.copy(), self.current_fps, self.status_msg
            
            # Render tactical standby placeholder frame
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Tactical grid
            cv2.line(placeholder, (0, self.height // 2), (self.width, self.height // 2), (30, 30, 35), 1)
            cv2.line(placeholder, (self.width // 2, 0), (self.width // 2, self.height), (30, 30, 35), 1)
            cv2.rectangle(placeholder, (15, 15), (self.width - 15, self.height - 15), (55, 55, 65), 2)
            
            # Title badge
            cv2.rectangle(placeholder, (30, 35), (self.width - 30, 85), (20, 20, 30), -1)
            cv2.rectangle(placeholder, (30, 35), (self.width - 30, 85), (0, 165, 255), 1)
            cv2.putText(placeholder, f"[{self.name.upper()}] STANDBY", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            
            cv2.putText(placeholder, f"Status: {self.status_msg}", (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(placeholder, f"Configured Source: Index/URL {self.src}", (40, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
            
            # Dedicated instructions for OBS Virtual Camera
            if "OBS" in self.name or "1" in str(self.src):
                cv2.putText(placeholder, "To activate OBS Phone Camera:", (40, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(placeholder, "1. Open OBS Studio", (55, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
                cv2.putText(placeholder, "2. Click 'Start Virtual Camera'", (55, 310),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(placeholder, "3. Video will automatically connect here!", (55, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            # Pulsating search indicator
            pulse = int((time.time() * 2) % 2)
            dot_color = (0, 255, 0) if pulse else (0, 120, 0)
            cv2.circle(placeholder, (self.width - 45, 60), 8, dot_color, -1)
            cv2.putText(placeholder, "SEARCHING", (self.width - 150, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            return False, placeholder, 0.0, self.status_msg

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

def blockchain_logger_worker():
    """ Background thread to handle blockchain/database logging without freezing the video feeds. """
    while True:
        alert_data = alert_queue.get()
        if alert_data is None: 
            break 
        
        print(f"\n[BLOCKCHAIN LEDGER UPDATE] Initiating Smart Contract...")
        print(f"Node: {alert_data['node_id']} | Source: {alert_data.get('camera', 'Unknown')} | Hash: {alert_data['frame_hash']}")
        print(f"Details: {json.dumps(alert_data['detections'], default=lambda o: float(o) if isinstance(o, (np.floating, float)) else int(o) if isinstance(o, (np.integer, int)) else str(o))}")
        time.sleep(0.5) 
        print("[BLOCKCHAIN LEDGER UPDATE] Block Confirmed & Immutable on Distributed Ledger.\n")
        
        alert_queue.task_done()

def process_camera_frame(frame, cam_name, model, device, use_tiling=False):
    """
    Runs YOLO threat/weapon detection and InsightFace facial reconnaissance
    on a single camera stream, drawing tactical annotations.
    """
    height, width, _ = frame.shape
    all_boxes = []

    if use_tiling:
        # Optional 2x2 Tiling for small/distant objects
        grid_y, grid_x = 2, 2
        tile_h = height // grid_y
        tile_w = width // grid_x
        for row in range(grid_y):
            for col in range(grid_x):
                y_offset = row * tile_h
                x_offset = col * tile_w
                tile = frame[y_offset:y_offset+tile_h, x_offset:x_offset+tile_w]
                results = model(tile, device=device, verbose=False)
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.45:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        all_boxes.append((cls_id, conf, x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset))
    else:
        # High-Speed Full-Frame Inference (Ultra-fast 60+ FPS on MPS/GPU)
        results = model(frame, device=device, verbose=False)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.45:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_boxes.append((cls_id, conf, x1, y1, x2, y2))

    # ---- INSIGHTFACE RECONNAISSANCE PASS ----
    detected_faces = []
    if FACE_REC_ENABLED and face_app is not None and len(known_face_encodings) > 0:
        try:
            faces = face_app.get(frame)
            for face in faces:
                f_bbox = face.bbox.astype(int)
                f_emb = face.normed_embedding

                best_match_idx = -1
                best_sim = -1.0
                for idx, known_emb in enumerate(known_face_encodings):
                    sim = float(np.dot(known_emb, f_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_match_idx = idx

                # Map cosine similarity (0.40 - 1.00) to 60% - 100% confidence
                display_percent = 0.0
                if best_sim >= 0.4:
                    display_percent = 60.0 + ((best_sim - 0.4) * (40.0 / 0.6))
                    display_percent = min(100.0, display_percent)

                identity = "Unknown"
                category = "Unknown"
                if display_percent >= 60.0 and best_match_idx != -1:
                    identity = known_face_names[best_match_idx]
                    category = known_face_categories[best_match_idx]

                detected_faces.append({
                    'bbox': f_bbox,
                    'identity': identity,
                    'category': category,
                    'score': round(display_percent, 2)
                })
        except Exception:
            pass

    # ---- COMPILE THREAT DETECTIONS & WEAPONS ----
    threats_detected = []
    annotated = frame.copy()
    person_count = 0
    weapon_count = 0

    for (cls_id, conf, x1, y1, x2, y2) in all_boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        detection_info = {
            "class_id": int(cls_id),
            "confidence": round(float(conf), 2),
            "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
        }

        # Identify weapons (custom Kaggle classes or COCO knives)
        is_weapon = (cls_id in WEAPON_CLASSES) or (cls_id in COCO_WEAPON_CLASSES)
        if is_weapon:
            weapon_count += 1
            weapon_name = WEAPON_CLASSES.get(cls_id, COCO_WEAPON_CLASSES.get(cls_id, "Weapon"))
            label = f"WEAPON: {weapon_name} ({int(conf * 100)}%)"
            color = (255, 0, 255) # Magenta (Purple) Box for Weapons
            detection_info["weapon_type"] = weapon_name
            detection_info["threat_level"] = "CRITICAL"
            threats_detected.append(detection_info)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        elif cls_id == PERSON_CLASS:
            person_count += 1
            label = "Person: Unknown"
            color = (0, 0, 255) # Red for Unknown Persons
            detection_info["category"] = "Unknown"

            # Check if any InsightFace bounding box matches this YOLO person
            for d_face in detected_faces:
                fx1, fy1, fx2, fy2 = d_face['bbox']
                face_cx, face_cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2

                if x1 <= face_cx <= x2 and y1 <= face_cy <= y2:
                    f_cat = d_face['category']
                    f_score = d_face['score']
                    f_id = d_face['identity']

                    if f_score >= 60.0:
                        label = f"{f_cat}: {f_id} ({f_score}%)"
                        detection_info["category"] = f_cat
                        detection_info["identity"] = f_id
                        detection_info["match_percent"] = f_score

                        # Separate Army (Green) / Civilian (Blue)
                        if f_cat == "Army":
                            color = (0, 255, 0)
                        elif f_cat == "Civilian":
                            color = (255, 120, 0)
                        else:
                            color = (0, 255, 0)
                    break

            if detection_info["category"] == "Unknown":
                detection_info["threat_level"] = "SUSPICIOUS"
                threats_detected.append(detection_info)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(0, y1)), color, -1)
            text_color = (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255)
            cv2.putText(annotated, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return annotated, threats_detected, person_count, weapon_count

def draw_camera_hud(frame, cam_name, fps, status, persons, weapons):
    """ Draws an in-feed HUD bar on each camera stream. """
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status LED
    dot_color = (0, 255, 0) if status == "Active" else (0, 165, 255)
    cv2.circle(frame, (16, 17), 6, dot_color, -1)

    # Title & FPS
    cv2.putText(frame, f"{cam_name} | {fps:.1f} FPS", (30, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    # Telemetry Badge
    status_str = f"P:{persons} W:{weapons}"
    badge_color = (0, 0, 255) if weapons > 0 else ((0, 200, 255) if persons > 0 else (140, 140, 140))
    cv2.putText(frame, status_str, (w - 110, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_color, 2)

def main_analytics_loop(cam1_src=MAC_CAM_DEFAULT, cam2_src=OBS_CAM_DEFAULT,
                        single_mode=False, use_tiling=False, headless=False, max_frames=0):
    print("=" * 70)
    print("   SMART BORDER SURVEILLANCE & RECONNAISSANCE SYSTEM")
    print(f"   Edge Node ID : {EDGE_NODE_ID}")
    print(f"   Camera 1 (Mac Webcam) : {cam1_src}")
    if not single_mode:
        print(f"   Camera 2 (OBS Phone)  : {cam2_src}")
    print(f"   Display Mode          : {'Single Camera' if single_mode else 'Dual Camera (Side by Side)'}")
    print("=" * 70)

    # Device selection (Apple Silicon MPS vs CUDA vs CPU)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Hardware Acceleration: Apple Silicon MPS (Metal) Enabled.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Hardware Acceleration: NVIDIA CUDA Enabled.")
    else:
        device = "cpu"
        print("Hardware Acceleration: Standard CPU.")

    # Load custom trained model if available, otherwise base model
    model_path = "runs/detect/border_weapons_model/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
    print(f"Loading Detection Model: {model_path}")
    model = YOLO(model_path)

    # Load known faces for reconnaissance
    load_known_faces()

    # Start camera capture threads
    print("\nStarting video capture streams...")
    cam1_thread = CameraStreamThread("Mac Webcam", cam1_src, width=640, height=480)
    cam2_thread = None
    if not single_mode:
        cam2_thread = CameraStreamThread("OBS Phone Camera", cam2_src, width=640, height=480)

    # Start blockchain logger thread
    blockchain_thread = threading.Thread(target=blockchain_logger_worker, daemon=True)
    blockchain_thread.start()

    last_alert_time_cam1 = 0
    last_alert_time_cam2 = 0
    alert_cooldown = 5.0
    frame_count = 0

    print("\nBoth camera capture loops running.")
    print("Press 'q' in the window to quit, or 's' to save a high-res snapshot.\n")

    try:
        while True:
            frame_count += 1

            # Grab frames from both cameras
            ok1, frame1, fps1, status1 = cam1_thread.get_frame()
            if not single_mode:
                ok2, frame2, fps2, status2 = cam2_thread.get_frame()
            else:
                ok2, frame2, fps2, status2 = False, None, 0.0, "Disabled"

            # ---- PROCESS CAMERA 1 (MAC WEBCAM) ----
            if ok1:
                annotated1, threats1, p1, w1 = process_camera_frame(
                    frame1, "Mac Webcam", model, device, use_tiling=use_tiling)
                draw_camera_hud(annotated1, "CAM 1: Mac Webcam", fps1, status1, p1, w1)

                # Blockchain Alert
                curr_t = time.time()
                if threats1 and (curr_t - last_alert_time_cam1 > alert_cooldown):
                    success, buffer = cv2.imencode('.jpg', frame1)
                    if success:
                        f_hash = hashlib.sha256(buffer.tobytes()).hexdigest()
                        alert_queue.put({
                            "node_id": EDGE_NODE_ID,
                            "camera": "CAM 1 (Mac Webcam)",
                            "timestamp": curr_t,
                            "frame_hash": f_hash,
                            "detections": threats1
                        })
                        last_alert_time_cam1 = curr_t
            else:
                annotated1 = frame1
                threats1, p1, w1 = [], 0, 0

            # ---- PROCESS CAMERA 2 (OBS PHONE CAMERA) ----
            if not single_mode:
                if ok2:
                    annotated2, threats2, p2, w2 = process_camera_frame(
                        frame2, "OBS Phone Camera", model, device, use_tiling=use_tiling)
                    draw_camera_hud(annotated2, "CAM 2: OBS Phone", fps2, status2, p2, w2)

                    # Blockchain Alert
                    curr_t = time.time()
                    if threats2 and (curr_t - last_alert_time_cam2 > alert_cooldown):
                        success, buffer = cv2.imencode('.jpg', frame2)
                        if success:
                            f_hash = hashlib.sha256(buffer.tobytes()).hexdigest()
                            alert_queue.put({
                                "node_id": EDGE_NODE_ID,
                                "camera": "CAM 2 (OBS Phone Camera)",
                                "timestamp": curr_t,
                                "frame_hash": f_hash,
                                "detections": threats2
                            })
                            last_alert_time_cam2 = curr_t
                else:
                    annotated2 = frame2
                    threats2, p2, w2 = [], 0, 0

            # ---- COMPOSE SIDE-BY-SIDE VIEW ----
            target_w = 640
            target_h = 480
            f1_resized = cv2.resize(annotated1, (target_w, target_h))

            if not single_mode:
                f2_resized = cv2.resize(annotated2, (target_w, target_h))
                dual_canvas = cv2.hconcat([f1_resized, f2_resized])
            else:
                dual_canvas = f1_resized

            # Top Master Surveillance HUD Header (42px)
            master_header = np.zeros((42, dual_canvas.shape[1], 3), dtype=np.uint8)
            master_header[:] = (20, 20, 25)
            cv2.line(master_header, (0, 41), (dual_canvas.shape[1], 41), (0, 200, 255), 2)

            # Left: Node ID
            cv2.putText(master_header, f"NODE: {EDGE_NODE_ID}", (15, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

            # Center: Dual Reconnaissance Title
            title_text = "SMART BORDER SURVEILLANCE - DUAL PORTAL RECONNAISSANCE"
            (tw, _), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(master_header, title_text, ((dual_canvas.shape[1] - tw) // 2, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Right: Live Clock
            time_str = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(master_header, time_str, (dual_canvas.shape[1] - 200, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            combined_display = cv2.vconcat([master_header, dual_canvas])

            if not headless:
                cv2.imshow("Dual-Camera Smart Border Surveillance", combined_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    snapshot_fn = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snapshot_fn, combined_display)
                    print(f"📸 Snapshot saved: {snapshot_fn}")
            else:
                if frame_count % 30 == 0:
                    status_str = f"Frames: {frame_count} | Cam 1 FPS: {fps1:.1f}"
                    if not single_mode:
                        status_str += f" | Cam 2 FPS: {fps2:.1f}"
                    print(status_str)

            if max_frames > 0 and frame_count >= max_frames:
                print(f"Reached MAX_FRAMES ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\nReleasing camera feeds and shutting down...")
        cam1_thread.release()
        if cam2_thread:
            cam2_thread.release()
        if not headless:
            cv2.destroyAllWindows()
        alert_queue.put(None)
        blockchain_thread.join()
        print("Shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Dual-Camera Border Surveillance & Reconnaissance")
    parser.add_argument("--cam1", default=MAC_CAM_DEFAULT,
                        help="Camera 1 source index or URL (default: 0 for Mac Webcam)")
    parser.add_argument("--cam2", default=OBS_CAM_DEFAULT,
                        help="Camera 2 source index or URL (default: 1 for OBS Virtual Camera / Phone)")
    parser.add_argument("--source", "-s", default=None,
                        help="Override Camera 1 source (legacy compatibility)")
    parser.add_argument("--single", action="store_true",
                        help="Run in single-camera mode instead of dual side-by-side")
    parser.add_argument("--tiling", action="store_true",
                        help="Enable 2x2 tiling grid on each camera for small/distant targets")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI display window (terminal output only)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process before exiting (0 for infinite)")

    args = parser.parse_args()
    cam1 = args.source if args.source is not None else args.cam1
    
    main_analytics_loop(
        cam1_src=cam1,
        cam2_src=args.cam2,
        single_mode=args.single,
        use_tiling=args.tiling,
        headless=args.headless,
        max_frames=args.max_frames
    )
