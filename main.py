import cv2
import numpy as np
import mediapipe as mp
import os
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
from cheat_sheet_detector import CheatSheetDetector
from cheating_state_machine import StudentStateMachine, CheatingDecisionEngine
from object_validator import ObjectValidator
from interaction_detector import InteractionDetector

# ============================================================
#  CONFIGURATION
# ============================================================
INPUT_VIDEO  = "input.mp4"
OUTPUT_VIDEO = "output_head_pose.mp4"

YOLO_CONF_DETECT  = 0.50
YOLO_CONF_TRACKER = 0.60

# Degree thresholds — tune by watching printed yaw/pitch values
YAW_LEFT_THRESHOLD   = -20
YAW_RIGHT_THRESHOLD  =  20
PITCH_DOWN_THRESHOLD = -18
PITCH_UP_THRESHOLD   =  18

# Ratio fallback thresholds
RATIO_YAW_LEFT   = 0.68
RATIO_YAW_RIGHT  = 0.32
RATIO_PITCH_DOWN = 0.66
RATIO_PITCH_UP   = 0.30

BACK_CONFIRM_FRAMES = 20
SMOOTH_BUFFER_LEN   = 10
HEAD_CROP_FRACTION  = 0.45
MIN_CROP_WIDTH      = 60
MIN_CROP_HEIGHT     = 50
STALE_TRACK_FRAMES  = 90

# ============================================================
#  FACE 3D MODEL  (normalised scale, not millimetres)
# ============================================================
FACE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

FACE_3D_MODEL = np.array([
    [ 0.000,  0.000,  0.000],
    [ 0.000, -0.330, -0.065],
    [-0.225,  0.170, -0.135],
    [ 0.225,  0.170, -0.135],
    [-0.150, -0.150, -0.125],
    [ 0.150, -0.150, -0.125],
], dtype=np.float64)

# ============================================================
#  COLORS
# ============================================================
DIR_COLOR = {
    "FORWARD": (0,   220,   0),
    "LEFT"   : (0,   165, 255),
    "RIGHT"  : (0,    80, 255),
    "DOWN"   : (0,     0, 220),
    "UP"     : (220, 220,   0),
    "BACK"   : (160,   0, 160),
    "UNKNOWN": (160, 160, 160),
}

# ============================================================
#  SMOOTHING BUFFER
# ============================================================
class SmoothedDirection:
    def __init__(self, maxlen=SMOOTH_BUFFER_LEN):
        self.buf = deque(maxlen=maxlen)

    def update(self, direction: str) -> str:
        self.buf.append(direction)
        return self.get()

    def get(self) -> str:
        real = [(d, i) for i, d in enumerate(self.buf) if d != "UNKNOWN"]
        if not real:
            return "UNKNOWN"
        n       = len(self.buf)
        weights = np.exp(np.linspace(0.0, 2.0, n))
        scores  = {}
        for d, idx in real:
            scores[d] = scores.get(d, 0.0) + weights[idx]
        return max(scores, key=scores.get)

# ============================================================
#  LANDMARK WRAPPER
# ============================================================
class LM:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z):
        self.x = x; self.y = y; self.z = z

# ============================================================
#  FIX 180° YAW AMBIGUITY
# ============================================================
def fix_yaw_flip(yaw: float, pitch: float, roll: float):
    """
    solvePnP sometimes returns the 'mirror' solution:
      yaw ≈ ±150° instead of yaw ≈ ∓30°
      roll ≈ ±170° instead of roll ≈ ∓10°

    Detection: if |yaw| > 90° AND |roll| > 90°, we have the flipped solution.
    Fix: subtract/add 180° to both yaw and roll, negate pitch.

    This is the standard correction for the solvePnP 180° ambiguity.
    """
    if abs(yaw) > 90 and abs(roll) > 90:
        # Flip to the correct solution
        yaw   = yaw   - np.sign(yaw)   * 180.0
        roll  = roll  - np.sign(roll)  * 180.0
        pitch = -pitch
    return yaw, pitch, roll


def angles_ok(yaw, pitch, roll) -> bool:
    """After flip correction, reject anything still out of human range."""
    return abs(yaw) <= 80 and abs(pitch) <= 80 and abs(roll) <= 70


# ============================================================
#  solvePnP HEAD POSE  (with flip correction + iterative hint)
# ============================================================
def estimate_head_pose(ff_lm: dict, img_w: int, img_h: int,
                       prev_rvec=None, prev_tvec=None):
    """
    Returns (yaw, pitch, roll, rvec, tvec) or (None, None, None, None, None).

    prev_rvec / prev_tvec: pass the previous frame's solution as a warm-start
    hint to SOLVEPNP_ITERATIVE. This dramatically reduces flip ambiguity
    because the solver starts near the previous (correct) solution.
    """
    try:
        pts_2d = np.array(
            [[ff_lm[i].x * img_w, ff_lm[i].y * img_h]
             for i in FACE_LANDMARK_IDS],
            dtype=np.float64,
        )
    except KeyError:
        return None, None, None, None, None

    f   = float(img_w)
    cam = np.array([[f, 0, img_w / 2.0],
                    [0, f, img_h / 2.0],
                    [0, 0, 1.0        ]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    use_hint = prev_rvec is not None and prev_tvec is not None

    ok, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, pts_2d, cam, dist,
        rvec=prev_rvec.copy() if use_hint else None,
        tvec=prev_tvec.copy() if use_hint else None,
        useExtrinsicGuess=use_hint,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rmat[2, 0],  sy))
        yaw   = np.degrees(np.arctan2( rmat[2, 1],  rmat[2, 2]))
        roll  = np.degrees(np.arctan2( rmat[1, 0],  rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2, 0],  sy))
        yaw   = np.degrees(np.arctan2(-rmat[1, 2],  rmat[1, 1]))
        roll  = 0.0

    # Apply 180° flip correction
    yaw, pitch, roll = fix_yaw_flip(yaw, pitch, roll)

    return yaw, pitch, roll, rvec, tvec


# ============================================================
#  RATIO FALLBACK
# ============================================================
def ratio_direction(ff_lm: dict) -> str:
    needed = [4, 10, 33, 152, 263]
    if not all(k in ff_lm for k in needed):
        return "UNKNOWN"

    nose     = ff_lm[4]
    forehead = ff_lm[10]
    l_eye    = ff_lm[33]
    chin     = ff_lm[152]
    r_eye    = ff_lm[263]

    eye_w  = abs(r_eye.x - l_eye.x)
    face_h = abs(chin.y - forehead.y)
    if eye_w < 1e-5 or face_h < 1e-5:
        return "UNKNOWN"

    yaw_r   = (nose.x - l_eye.x) / eye_w
    pitch_r = (nose.y - forehead.y) / face_h

    if pitch_r > RATIO_PITCH_DOWN:
        return "DOWN"
    if pitch_r < RATIO_PITCH_UP:
        return "UP"
    if yaw_r > RATIO_YAW_LEFT:
        return "LEFT"
    if yaw_r < RATIO_YAW_RIGHT:
        return "RIGHT"
    return "FORWARD"


# ============================================================
#  CLASSIFY ANGLES
# ============================================================
def classify(yaw: float, pitch: float) -> str:
    if pitch < PITCH_DOWN_THRESHOLD:
        return "DOWN"
    if pitch > PITCH_UP_THRESHOLD:
        return "UP"
    if yaw   < YAW_LEFT_THRESHOLD:
        return "LEFT"
    if yaw   > YAW_RIGHT_THRESHOLD:
        return "RIGHT"
    return "FORWARD"


# ============================================================
#  INIT MODELS
# ============================================================
print("[INFO] Loading YOLO …")
yolo = YOLO("yolov8n.pt")

print("[INFO] Initialising DeepSORT …")
tracker = DeepSort(
    max_age=90, n_init=5,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
    max_iou_distance=0.7,
    embedder="mobilenet",
    half=True,
)

print("[INFO] Initialising MediaPipe …")
_mp_face = mp.solutions.face_mesh
_mp_pose = mp.solutions.pose

face_mesh = _mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
pose_est = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================
#  TRACK STATE
# ============================================================
baseline_path = "baseline.json"
baseline = None
if os.path.exists(baseline_path):
    try:
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
        print(f"[main.py] Loaded baseline parameters from {baseline_path}")
    except Exception as e:
        print(f"[main.py] Failed to load baseline: {e}")

track_db: dict = {}
detector = CheatSheetDetector(baseline_path=baseline_path)
object_validator = ObjectValidator()
cheating_db = {}
decision_engine = CheatingDecisionEngine()
interaction_detector = InteractionDetector()

def new_state():
    return {
        "smoother"     : SmoothedDirection(),
        "confirmed"    : "UNKNOWN",
        "no_face_count": 0,
        "seen_face"    : False,
        "last_frame"   : 0,
        # Store previous solvePnP solution as warm-start for next frame
        "prev_rvec"    : None,
        "prev_tvec"    : None,
    }

# ============================================================
#  OPEN VIDEO
# ============================================================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {INPUT_VIDEO}")

W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS   = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS, (W, H),
)

print(f"[INFO] {W}x{H} @ {FPS} fps | {TOTAL} frames")
print("[INFO] Running … press Q to quit.\n")

fidx = 0

# ============================================================
#  MAIN LOOP
# ============================================================
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    fidx += 1
    
    sec = fidx / FPS
    timestamp_str = f"[{int(sec)//60:02d}:{sec%60:05.2f}]"
    
    active_students_frame = {}
    validated_sheets_frame = []

    # ── YOLO ─────────────────────────────────────────────────
    dets = []
    for r in yolo(frame, classes=[0], conf=YOLO_CONF_DETECT, verbose=False):
        for box in r.boxes:
            c = float(box.conf[0])
            if c < YOLO_CONF_TRACKER:
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            dets.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], c, "person"))

    # ── DeepSORT ─────────────────────────────────────────────
    tracks = tracker.update_tracks(dets, frame=frame)

    # ── MediaPipe Processing Pass ─────────────────────────────
    track_info = {}
    for track in tracks:
        if not track.is_confirmed():
            continue

        tid          = track.track_id
        x1,y1,x2,y2 = map(int, track.to_ltrb())
        x1,y1        = max(0,x1), max(0,y1)
        x2,y2        = min(W,x2), min(H,y2)
        if x2 <= x1 or y2 <= y1:
            continue

        if tid not in track_db:
            track_db[tid] = new_state()
        st = track_db[tid]
        st["last_frame"] = fidx

        ph   = y2 - y1
        pw   = x2 - x1
        hy2  = y1 + int(ph * HEAD_CROP_FRACTION)
        hpad = int(pw * 0.08)

        cx1 = max(0, x1 - hpad)
        cx2 = min(W, x2 + hpad)
        cy1 = y1
        cy2 = min(H, hy2)

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        raw    = "UNKNOWN"
        log_extra = ""

        person_crop = frame[y1:y2, x1:x2]
        pose_res = None
        if person_crop.size > 0:
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_res = pose_est.process(person_rgb)

        nose_coords = None
        if crop_w >= MIN_CROP_WIDTH and crop_h >= MIN_CROP_HEIGHT:
            crop_rgb = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
            face_res = face_mesh.process(crop_rgb)

            if face_res.multi_face_landmarks:
                st["no_face_count"] = 0
                st["seen_face"]     = True
                lm = face_res.multi_face_landmarks[0].landmark

                all_ids = set(FACE_LANDMARK_IDS) | {4, 10, 33, 133, 152, 263, 362}
                ff_lm = {}
                for idx in all_ids:
                    if idx >= len(lm):
                        continue
                    px = lm[idx].x * crop_w + cx1
                    py = lm[idx].y * crop_h + cy1
                    ff_lm[idx] = LM(px / W, py / H, lm[idx].z)
                
                if 4 in ff_lm:
                    nose_coords = (int(ff_lm[4].x * W), int(ff_lm[4].y * H))

                yaw, pitch, roll, rvec, tvec = estimate_head_pose(
                    ff_lm, W, H,
                    prev_rvec=st["prev_rvec"],
                    prev_tvec=st["prev_tvec"],
                )

                if yaw is not None and angles_ok(yaw, pitch, roll):
                    st["prev_rvec"] = rvec
                    st["prev_tvec"] = tvec
                    raw = classify(yaw, pitch)
                else:
                    st["prev_rvec"] = None
                    st["prev_tvec"] = None
                    raw = ratio_direction(ff_lm)
            else:
                st["no_face_count"] += 1
                st["prev_rvec"] = None
                st["prev_tvec"] = None

                is_back = False
                if pose_res and pose_res.pose_landmarks:
                    pl    = pose_res.pose_landmarks.landmark
                    PL    = _mp_pose.PoseLandmark
                    l_vis = pl[PL.LEFT_SHOULDER].visibility
                    r_vis = pl[PL.RIGHT_SHOULDER].visibility
                    n_vis = pl[PL.NOSE].visibility
                    if l_vis > 0.5 and r_vis > 0.5 and n_vis < 0.3:
                        is_back = True

                if is_back:
                    raw = "BACK"
                elif st["seen_face"] and st["no_face_count"] >= BACK_CONFIRM_FRAMES:
                    raw = "BACK"

        direction       = st["smoother"].update(raw)
        st["confirmed"] = direction

        lw_x, lw_y, rw_x, rw_y = None, None, None, None
        lh_x, lh_y, rh_x, rh_y = None, None, None, None
        hand_near_pocket = False
        wrists_low = False

        if pose_res and pose_res.pose_landmarks:
            pl = pose_res.pose_landmarks.landmark
            PL = _mp_pose.PoseLandmark

            lw = pl[PL.LEFT_WRIST]
            rw = pl[PL.RIGHT_WRIST]
            lh = pl[PL.LEFT_HIP]
            rh = pl[PL.RIGHT_HIP]

            lw_x, lw_y = int(lw.x * pw + x1), int(lw.y * ph + y1)
            rw_x, rw_y = int(rw.x * pw + x1), int(rw.y * ph + y1)
            lh_x, lh_y = int(lh.x * pw + x1), int(lh.y * ph + y1)
            rh_x, rh_y = int(rh.x * pw + x1), int(rh.y * ph + y1)

            dist_l = np.sqrt((lw_x - lh_x)**2 + (lw_y - lh_y)**2)
            dist_r = np.sqrt((rw_x - rh_x)**2 + (rw_y - rh_y)**2)
            
            pocket_ratio = 0.12
            if baseline:
                wrist_mvmt = baseline.get("writing_wrist_movement", {})
                pocket_ratio = wrist_mvmt.get("pocket_proximity_thresh_ratio", 0.12)
                
            pocket_thresh = ph * pocket_ratio
            if dist_l < pocket_thresh or dist_r < pocket_thresh:
                hand_near_pocket = True

            if (lw_y > y1 + ph * 0.4) or (rw_y > y1 + ph * 0.4):
                wrists_low = True

        track_info[tid] = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "pw": pw, "ph": ph,
            "person_crop": person_crop,
            "wrists": ((lw_x, lw_y) if lw_x is not None else None,
                       (rw_x, rw_y) if rw_x is not None else None),
            "hips": ((lh_x, lh_y) if lh_x is not None else None,
                     (rh_x, rh_y) if rh_x is not None else None),
            "hand_near_pocket": hand_near_pocket,
            "wrists_low": wrists_low,
            "nose_coords": nose_coords,
            "direction": direction
        }

    # ── Interaction & Sheet Detection Pass ────────────────────
    active_students_frame = {}
    validated_sheets_frame = []

    tids = list(track_info.keys())
    for i, tid in enumerate(tids):
        info = track_info[tid]
        
        # Calculate if this student is interacting with any other student
        is_interacting = False
        lw_A, rw_A = info["wrists"]
        diag_A = np.sqrt(info["pw"]**2 + info["ph"]**2)
        
        for j, other_tid in enumerate(tids):
            if other_tid == tid:
                continue
            other_info = track_info[other_tid]
            lw_B, rw_B = other_info["wrists"]
            
            wrists_A = [w for w in (lw_A, rw_A) if w is not None]
            wrists_B = [w for w in (lw_B, rw_B) if w is not None]
            if wrists_A and wrists_B:
                min_hand_dist = min(np.sqrt((wA[0]-wB[0])**2 + (wA[1]-wB[1])**2) for wA in wrists_A for wB in wrists_B)
                diag_B = np.sqrt(other_info["pw"]**2 + other_info["ph"]**2)
                avg_diag = (diag_A + diag_B) / 2.0
                hand_close_thresh = avg_diag * 0.25
                if min_hand_dist < hand_close_thresh:
                    is_interacting = True
                    break
        
        detected_sheets = detector.detect(
            info["person_crop"],
            info["x1"], info["y1"],
            info["pw"], info["ph"],
            info["wrists"],
            is_interacting=is_interacting
        )
        chit_detected = len(detected_sheets) > 0
        
        paper_conf = 0.0
        bottle_conf = 0.0
        obj_center = None
        if chit_detected:
            first_sheet = detected_sheets[0]
            paper_conf = first_sheet.get('paper_conf', 0.0)
            bottle_conf = first_sheet.get('bottle_conf', 0.0)
            obj_center = first_sheet.get('center', None)

        for sheet in detected_sheets:
            validated_sheets_frame.append({
                "sheet_id": id(sheet),
                "bbox": sheet.get("box", (0, 0, 0, 0)),
                "center": sheet.get("center", (0, 0)),
                "confidence": sheet.get("confidence", 0.0),
                "detected_by_tid": tid,
                "paper_conf": sheet.get("paper_conf", 0.0),
                "bottle_conf": sheet.get("bottle_conf", 0.0)
            })

        # Update student state machine using baseline parameters
        if tid not in cheating_db:
            cheating_db[tid] = StudentStateMachine(tid, baseline=baseline)
        c_st = cheating_db[tid]
        c_st.update(chit_detected, info["direction"], info["wrists_low"], info["hand_near_pocket"], timestamp_str,
                    paper_conf, bottle_conf, obj_center, info["nose_coords"], info["ph"])

        is_cheating, _ = decision_engine.evaluate(c_st)
        active_students_frame[tid] = {
            "bbox": (info["x1"], info["y1"], info["x2"], info["y2"]),
            "wrists": info["wrists"],
            "hips": info["hips"],
            "head_center": info["nose_coords"],
            "direction": info["direction"],
            "pw": info["pw"],
            "ph": info["ph"],
            "confirmed_cheating": is_cheating
        }

        # ── Draw ─────────────────────────────────────────────
        col   = DIR_COLOR.get(info["direction"], (200, 200, 200))
        label = f"ID:{tid}  {info['direction']}"

        cv2.rectangle(frame, (info["x1"], info["y1"]), (info["x2"], info["y2"]), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        lx = info["x1"]
        ly = max(info["y1"] - 6, th + 6)
        cv2.rectangle(frame, (lx, ly-th-4), (lx+tw+6, ly+2), col, cv2.FILLED)
        cv2.putText(frame, label, (lx+3, ly-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # ── Cheat Sheet Passing Detection ────────────────────────
    confirmed_passing_pairs = interaction_detector.update(
        active_students_frame,
        validated_sheets_frame,
        timestamp_str,
        fidx
    )

    # ── Draw Cheating Warning Banners ────────────────────────
    alert_y = 25
    for tid, c_st in list(cheating_db.items()):
        is_cheating, reason = decision_engine.evaluate(c_st)
        if is_cheating:
            lines = [
                "ALERT: CHEATING DETECTED",
                f"Student ID: {tid}",
                f"Reason: {reason}"
            ]
            max_w = 0
            total_h = 0
            line_heights = []
            for line in lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                max_w = max(max_w, tw)
                line_heights.append(th)
                total_h += th + 6
                
            # Draw translucent backdrop or solid red rectangle
            cv2.rectangle(frame, (10, alert_y), (20 + max_w, alert_y + total_h + 10), (0, 0, 255), cv2.FILLED)
            curr_y = alert_y + 18
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (15, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                curr_y += line_heights[i] + 6
            alert_y += total_h + 20

    for pair in confirmed_passing_pairs:
        tid_A, tid_B = pair
        lines = [
            "CHEAT SHEET PASSING DETECTED",
            f"Student A : {tid_A}",
            f"Student B : {tid_B}",
            "Reason: Unauthorized transfer of cheat sheet."
        ]
        max_w = 0
        total_h = 0
        line_heights = []
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            max_w = max(max_w, tw)
            line_heights.append(th)
            total_h += th + 6
            
        cv2.rectangle(frame, (10, alert_y), (20 + max_w, alert_y + total_h + 10), (0, 0, 255), cv2.FILLED)
        curr_y = alert_y + 18
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (15, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            curr_y += line_heights[i] + 6
        alert_y += total_h + 20

    # ── Frame counter ────────────────────────────────────────
    cv2.putText(frame, f"{fidx}/{TOTAL}", (8, H-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Stale track cleanup ──────────────────────────────────
    for t in [t for t, s in list(track_db.items())
              if fidx - s["last_frame"] > STALE_TRACK_FRAMES]:
        del track_db[t]
        if t in cheating_db:
            del cheating_db[t]

    out.write(frame)
    cv2.imshow("Head Pose Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
#  CLEANUP
# ============================================================
cap.release()
out.release()
cv2.destroyAllWindows()
face_mesh.close()
pose_est.close()

print(f"\n[DONE] -> {OUTPUT_VIDEO}")