import cv2
import numpy as np
import json
import os
from object_validator import ObjectValidator

class CheatSheetDetector:
    """
    A modular class responsible for detecting cheat sheets in a person's bounding box crop.
    Loads baseline.json to ignore static desk papers and utilizes scale-invariant relative thresholds.
    Integrates ObjectValidator to compute paper and bottle confidences for each candidate.
    """
    def __init__(self, baseline_path="baseline.json"):
        self.baseline = None
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, "r") as f:
                    self.baseline = json.load(f)
                print(f"[CheatSheetDetector] Loaded baseline from {baseline_path}")
            except Exception as e:
                print(f"[CheatSheetDetector] Warning: failed to load baseline: {e}")
                
        # Color thresholds for white/off-white paper
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 50, 255])
        self.validator = ObjectValidator()
        
    def detect(self, person_crop, tx1, ty1, pw, ph, wrists, is_interacting=False):
        """
        Detects newly appearing cheat sheets inside the student crop.
        
        person_crop: BGR crop of the person bounding box.
        tx1, ty1: Absolute offset of the person crop top-left corner in full frame.
        pw, ph: Person width and height.
        wrists: Tuple of (left_wrist, right_wrist) absolute frame coordinates.
        is_interacting: Whether the student is in proximity with another student.
        
        Returns:
            list of dict: Suspicious new objects matching cheat sheet criteria.
        """
        detected_sheets = []
        if person_crop.size == 0:
            return detected_sheets

        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Scale-Invariant Relative Thresholds
        diag = np.sqrt(pw**2 + ph**2)
        wrist_dist_thresh = diag * (0.35 if is_interacting else 0.15)  # Hand proximity: 35% of bounding box diagonal
        
        # Dimensions relative to student bounding box
        min_w = max(4, int(pw * 0.04))
        max_w = int(pw * 0.35)
        min_h = max(4, int(ph * 0.03))
        max_h = int(ph * 0.25)
        min_area = int(pw * ph * 0.001)
        max_area = int(pw * ph * 0.04)

        lw, rw = wrists
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            # Filter by relative dimensions and area bounds
            if min_w < w < max_w and min_h < h < max_h and min_area < area < max_area:
                cx = x + w // 2 + tx1
                cy = y + h // 2 + ty1
                
                # Change Detection filter: Ignore if matches baseline desk papers
                is_baseline = False
                if self.baseline:
                    for obj in self.baseline.get("static_desk_objects", []):
                        ox1, oy1, ox2, oy2 = obj["bbox"]
                        pad = 20
                        if (ox1 - pad <= cx <= ox2 + pad) and (oy1 - pad <= cy <= oy2 + pad):
                            base_area = obj.get("area", 9999)
                            # If the detected contour is much smaller than the baseline paper,
                            # it's a new cheat sheet on top of it.
                            if area > 0.65 * base_area:
                                is_baseline = True
                                break
                            
                if is_baseline:
                    continue  # Ignore pre-existing desk items
                
                # Check hand proximity
                near_wrist = False
                if lw is not None:
                    d_l = np.sqrt((cx - lw[0])**2 + (cy - lw[1])**2)
                    if d_l < wrist_dist_thresh:
                        near_wrist = True
                if rw is not None:
                    d_r = np.sqrt((cx - rw[0])**2 + (cy - rw[1])**2)
                    if d_r < wrist_dist_thresh:
                        near_wrist = True
                
                if near_wrist:
                    # Check vertical position: must be below upper body (lap/desk level)
                    if cy > ty1 + ph * 0.4:
                        # Run shape classification validation
                        res = self.validator.validate(cnt, pw, ph)
                        detected_sheets.append({
                            'box': (x + tx1, y + ty1, x + w + tx1, y + h + ty1),
                            'center': (cx, cy),
                            'confidence': 1.0,
                            'paper_conf': res["paper"],
                            'bottle_conf': res["bottle"],
                            'unknown_conf': res["unknown"]
                        })
                        
        return detected_sheets
