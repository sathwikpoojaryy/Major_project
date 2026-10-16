import cv2
import numpy as np
# pyrefly: ignore [missing-import]
import mediapipe as mp
import json
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class BaselineLearner:
    """
    Learns normal exam behavior and environment statistics from a baseline video (input_normal.mp4).
    Generates a structured baseline.json file containing:
      - Static desk objects (bounding boxes, areas, aspect ratios of answer sheets, etc.)
      - Normal writing posture (mean and std of gaze pitch/yaw)
      - Normal wrist movement parameters (minimum wrist-to-hip height ratio)
    """
    def __init__(self, model_dir=r"c:\Users\Sahana\Desktop\final year project\project 1"):
        self.model_dir = model_dir
        self.yolo_path = os.path.join(model_dir, "yolov8n.pt")
        
        # Color thresholds for off-white paper
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 50, 255])
        
        # Face PnP reference model
        self.FACE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]
        self.FACE_3D_MODEL = np.array([
            [ 0.000,  0.000,  0.000],
            [ 0.000, -0.330, -0.065],
            [-0.225,  0.170, -0.135],
            [ 0.225,  0.170, -0.135],
            [-0.150, -0.150, -0.125],
            [ 0.150, -0.150, -0.125],
        ], dtype=np.float64)

    def _fix_yaw_flip(self, yaw: float, pitch: float, roll: float):
        if abs(yaw) > 90 and abs(roll) > 90:
            yaw   = yaw   - np.sign(yaw)   * 180.0
            roll  = roll  - np.sign(roll)  * 180.0
            pitch = -pitch
        return yaw, pitch, roll

    def _estimate_head_pose(self, ff_lm: dict, img_w: int, img_h: int):
        try:
            pts_2d = np.array([[ff_lm[i].x * img_w, ff_lm[i].y * img_h] for i in self.FACE_LANDMARK_IDS], dtype=np.float64)
        except KeyError:
            return None, None
        f = float(img_w)
        cam = np.array([[f, 0, img_w / 2.0], [0, f, img_h / 2.0], [0, 0, 1.0]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(self.FACE_3D_MODEL, pts_2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None
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
        yaw, pitch, roll = self._fix_yaw_flip(yaw, pitch, roll)
        return yaw, pitch

    def learn(self, video_path, output_json_path):
        print(f"[BaselineLearner] Loading models and opening video: {video_path}...")
        yolo = YOLO(self.yolo_path)
        tracker = DeepSort(max_age=90, n_init=5, embedder="mobilenet", half=True)
        
        mp_pose = mp.solutions.pose
        pose_est = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open baseline video: {video_path}")
            
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Aggregation structures
        gaze_pitches = []
        gaze_yaws = []
        wrist_hip_ratios = []
        
        # Bounding box clusters of detected white sheets:
        # Format: { 'id': int, 'bbox': [x1, y1, x2, y2], 'area': float, 'aspect_ratio': float, 'count': int }
        paper_candidates = []
        candidate_id_counter = 0

        fidx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            fidx += 1

            dets = []
            for r in yolo(frame, classes=[0], conf=0.5, verbose=False):
                for box in r.boxes:
                    c = float(box.conf[0])
                    if c < 0.6:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    dets.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], c, "person"))

            tracks = tracker.update_tracks(dets, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
                tx1, ty1 = max(0, tx1), max(0, ty1)
                tx2, ty2 = min(W, tx2), min(H, ty2)
                pw, ph = tx2 - tx1, ty2 - ty1

                person_crop = frame[ty1:ty2, tx1:tx2]
                if person_crop.size == 0:
                    continue

                # 1. Pose estimation for wrists and hips
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_res = pose_est.process(person_rgb)

                if pose_res and pose_res.pose_landmarks:
                    pl = pose_res.pose_landmarks.landmark
                    PL = mp_pose.PoseLandmark
                    lw = pl[PL.LEFT_WRIST]
                    rw = pl[PL.RIGHT_WRIST]
                    lh = pl[PL.LEFT_HIP]
                    rh = pl[PL.RIGHT_HIP]

                    # Scale to absolute coordinates
                    lw_x, lw_y = int(lw.x * pw + tx1), int(lw.y * ph + ty1)
                    rw_x, rw_y = int(rw.x * pw + tx1), int(rw.y * ph + ty1)
                    lh_x, lh_y = int(lh.x * pw + tx1), int(lh.y * ph + ty1)
                    rh_x, rh_y = int(rh.x * pw + tx1), int(rh.y * ph + ty1)

                    dist_l = np.sqrt((lw_x - lh_x)**2 + (lw_y - lh_y)**2)
                    dist_r = np.sqrt((rw_x - rh_x)**2 + (rw_y - rh_y)**2)
                    
                    # Store distances normalized by body height
                    wrist_hip_ratios.append(dist_l / ph)
                    wrist_hip_ratios.append(dist_r / ph)

                # 2. Gaze estimation
                hy2 = ty1 + int(ph * 0.45)
                hpad = int(pw * 0.08)
                cx1 = max(0, tx1 - hpad)
                cx2 = min(W, tx2 + hpad)
                cy1 = ty1
                cy2 = min(H, hy2)
                crop_w = cx2 - cx1
                crop_h = cy2 - cy1

                if crop_w >= 60 and crop_h >= 50:
                    crop_rgb = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
                    face_res = face_mesh.process(crop_rgb)
                    if face_res.multi_face_landmarks:
                        lm = face_res.multi_face_landmarks[0].landmark
                        all_ids = set(self.FACE_LANDMARK_IDS) | {4, 10, 33, 133, 152, 263, 362}
                        ff_lm = {}
                        for idx in all_ids:
                            if idx >= len(lm):
                                continue
                            px = lm[idx].x * crop_w + cx1
                            py = lm[idx].y * crop_h + cy1
                            # Define dummy landmark wrapper inline
                            class LM_wrap:
                                def __init__(self, x, y, z):
                                    self.x = x; self.y = y; self.z = z
                            ff_lm[idx] = LM_wrap(px / W, py / H, lm[idx].z)
                            
                        y, p = self._estimate_head_pose(ff_lm, W, H)
                        if y is not None:
                            gaze_pitches.append(p)
                            gaze_yaws.append(y)

                # 3. Detect all desk papers (HSV white contours) in student crop
                hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)
                    
                    # Acceptable paper-like size bounds relative to person size
                    # Paper width must be at least 1% of frame size and area must be substantial
                    if 15 < w < 250 and 15 < h < 250 and area > 100:
                        ax1, ay1 = x + tx1, y + ty1
                        ax2, ay2 = x + w + tx1, y + h + ty1
                        acx, acy = (ax1 + ax2) // 2, (ay1 + ay2) // 2
                        ar = float(w) / h if h > 0 else 1.0
                        
                        # Match with existing candidates
                        matched = False
                        for cand in paper_candidates:
                            # If centers overlap within 40 pixels, merge
                            ccx, ccy = (cand['bbox'][0] + cand['bbox'][2]) // 2, (cand['bbox'][1] + cand['bbox'][3]) // 2
                            dist = np.sqrt((acx - ccx)**2 + (acy - ccy)**2)
                            if dist < 40:
                                # Update candidate average bounding box and features
                                cand['bbox'] = [
                                    int(cand['bbox'][0] * 0.9 + ax1 * 0.1),
                                    int(cand['bbox'][1] * 0.9 + ay1 * 0.1),
                                    int(cand['bbox'][2] * 0.9 + ax2 * 0.1),
                                    int(cand['bbox'][3] * 0.9 + ay2 * 0.1)
                                ]
                                cand['area'] = cand['area'] * 0.9 + area * 0.1
                                cand['aspect_ratio'] = cand['aspect_ratio'] * 0.9 + ar * 0.1
                                cand['count'] += 1
                                matched = True
                                break
                                
                        if not matched:
                            paper_candidates.append({
                                'id': candidate_id_counter,
                                'bbox': [ax1, ay1, ax2, ay2],
                                'area': float(area),
                                'aspect_ratio': float(ar),
                                'count': 1
                            })
                            candidate_id_counter += 1

            if fidx % 20 == 0:
                print(f"[BaselineLearner] Processed {fidx}/{total_frames} frames...")

        cap.release()
        pose_est.close()
        face_mesh.close()

        # Filtering: select papers that appear consistently (e.g. in at least 15% of frames)
        static_desk_objects = []
        min_appearance_frames = max(5, int(fidx * 0.12))
        
        for cand in paper_candidates:
            if cand['count'] >= min_appearance_frames:
                static_desk_objects.append({
                    "label": f"static_paper_{cand['id']}",
                    "bbox": cand['bbox'],
                    "area": float(cand['area']),
                    "aspect_ratio": float(cand['aspect_ratio'])
                })
        
        # Posture Stats
        pitch_mean = float(np.mean(gaze_pitches)) if gaze_pitches else -25.0
        pitch_std  = float(np.std(gaze_pitches)) if gaze_pitches else 5.0
        yaw_mean   = float(np.mean(gaze_yaws)) if gaze_yaws else 0.0
        yaw_std    = float(np.std(gaze_yaws)) if gaze_yaws else 10.0
        
        # Wrist Distances
        if wrist_hip_ratios:
            # We take the 10th percentile as the minimum normal writing range to avoid outliers
            wrist_hip_dist_min_ratio = float(np.percentile(wrist_hip_ratios, 10))
        else:
            wrist_hip_dist_min_ratio = 0.22
            
        # The threshold to trigger pocket reach is set slightly below the writing minimum ratio
        pocket_proximity_thresh_ratio = float(max(0.12, wrist_hip_dist_min_ratio * 0.75))

        baseline_profile = {
            "static_desk_objects": static_desk_objects,
            "writing_posture": {
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "yaw_mean": yaw_mean,
                "yaw_std": yaw_std
            },
            "writing_wrist_movement": {
                "min_wrist_to_hip_ratio": wrist_hip_dist_min_ratio,
                "pocket_proximity_thresh_ratio": pocket_proximity_thresh_ratio
            }
        }
        
        # Save to JSON file
        with open(output_json_path, "w") as f:
            json.dump(baseline_profile, f, indent=4)
            
        print(f"[BaselineLearner] Saved baseline profile containing {len(static_desk_objects)} static objects to {output_json_path}")
        return baseline_profile
