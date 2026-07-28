import numpy as np

class StudentStateMachine:
    """
    Temporal State Machine tracking cheating sequences:
    NORMAL -> POCKET_REACH -> UNKNOWN_OBJECT_APPEARS -> CHEAT_SHEET_HELD ->
    READING_CHEAT_SHEET -> CHEAT_SHEET_UNDER_ANSWER_SHEET -> RETURN_TO_POCKET -> CHEATING_CONFIRMED
    """
    def __init__(self, student_id, baseline=None):
        self.student_id = student_id
        self.state = "NORMAL"
        self.state_history = []
        self.baseline = baseline
        
        # Configuration ratios
        self.pocket_proximity_thresh_ratio = 0.12
        if baseline:
            wrist_movement = baseline.get("writing_wrist_movement", {})
            self.pocket_proximity_thresh_ratio = wrist_movement.get("pocket_proximity_thresh_ratio", 0.12)

        # Counter thresholds and frames
        self.frame_counter = 0
        self.consecutive_contour_frames = 0
        self.consecutive_down_gaze = 0
        self.under_sheet_frames = 0
        self.return_pocket_frames = 0
        self.cheat_sheet_lost_frames = 0
        
        # Pocket reach motion tracker
        self.pocket_reach_start_frame = -1
        self.pocket_reach_low_count = 0
        self.pocket_reach_motion_confirmed = False
        
        # Return to pocket motion tracker
        self.return_pocket_start_frame = -1
        self.return_pocket_low_count = 0
        self.return_pocket_motion_confirmed = False

        # History buffers
        self.gaze_history = []
        self.contour_history = []
        self.hand_pocket_history = []

        # Trajectory history (visited regions)
        self.visited_regions = []
        self.trajectory_path = [] # Running log of regions
        
        # Evidence scorecard
        self.score_card = {
            "pocket_reach": 0,
            "paper_geometry": 0,
            "follows_hand": 0,
            "down_gaze": 0,
            "hidden_under_paper": 0,
            "returned_to_pocket": 0,
            "bottle_geometry": 0,
            "moved_to_mouth": 0
        }

    def get_evidence_score(self) -> int:
        return sum(self.score_card.values())

    def update(self, chit_detected: bool, gaze: str, wrists_low: bool, hand_near_pocket: bool, 
               timestamp_str: str, paper_conf: float, bottle_conf: float, 
               obj_center: tuple, nose_coords: tuple, ph: int):
        """
        Updates the state transition path and tracks object trajectories and scoring.
        """
        self.frame_counter += 1
        prev_state = self.state
        reason = ""

        # 1. Track object region trajectory
        region = "DESK"
        if chit_detected and obj_center is not None:
            # Check if close to nose (mouth area)
            if nose_coords is not None:
                dist_nose = np.sqrt((obj_center[0] - nose_coords[0])**2 + (obj_center[1] - nose_coords[1])**2)
                if dist_nose < ph * 0.22: # Close to mouth (22% of body height)
                    region = "MOUTH"
                    self.score_card["moved_to_mouth"] = -10
            
            # If not mouth, check if close to pocket/hip
            if region != "MOUTH" and hand_near_pocket:
                region = "POCKET"
            elif region != "MOUTH":
                region = "HAND"
        elif self.state == "CHEAT_SHEET_UNDER_ANSWER_SHEET":
            region = "UNDER_PAPER"

        # Record visited region sequence
        if region not in self.visited_regions:
            self.visited_regions.append(region)
        self.trajectory_path.append(region)
        if len(self.trajectory_path) > 100:
            self.trajectory_path.pop(0)

        # 2. Track actual pocket reaching motion
        if hand_near_pocket:
            if self.pocket_reach_start_frame == -1:
                self.pocket_reach_start_frame = self.frame_counter
                self.pocket_reach_low_count = 1
            else:
                self.pocket_reach_low_count += 1
        else:
            if self.pocket_reach_start_frame != -1:
                # Wrist has left the pocket area
                if 2 <= self.pocket_reach_low_count <= 45:
                    self.pocket_reach_motion_confirmed = True
                    self.score_card["pocket_reach"] = 3
                self.pocket_reach_start_frame = -1
                self.pocket_reach_low_count = 0

        # Track stowing return to pocket motion
        if self.state == "CHEAT_SHEET_UNDER_ANSWER_SHEET" and hand_near_pocket:
            if self.return_pocket_start_frame == -1:
                self.return_pocket_start_frame = self.frame_counter
                self.return_pocket_low_count = 1
            else:
                self.return_pocket_low_count += 1
        else:
            if self.return_pocket_start_frame != -1:
                if 2 <= self.return_pocket_low_count <= 45:
                    self.return_pocket_motion_confirmed = True
                    self.score_card["returned_to_pocket"] = 3
                self.return_pocket_start_frame = -1
                self.return_pocket_low_count = 0

        # Update scoring on geometry
        if chit_detected:
            if paper_conf > 0.65:
                self.score_card["paper_geometry"] = 4
            if bottle_conf > 0.65:
                self.score_card["bottle_geometry"] = -6

        # FSM State Transitions
        if self.state == "NORMAL":
            if self.pocket_reach_motion_confirmed:
                self.state = "POCKET_REACH"
                reason = "Pocket reach motion confirmed (wrist moved to hip, dwelled, and left)."
                self.pocket_reach_motion_confirmed = False
                
        elif self.state == "POCKET_REACH":
            if chit_detected:
                self.state = "UNKNOWN_OBJECT_APPEARS"
                self.consecutive_contour_frames = 1
                self.cheat_sheet_lost_frames = 0
                reason = "Unknown object appeared near the hand."
            else:
                # Timeout back to normal
                if self.frame_counter - self.pocket_reach_start_frame > 60:
                    self.state = "NORMAL"
                    self.pocket_reach_motion_confirmed = False
                    
        elif self.state == "UNKNOWN_OBJECT_APPEARS":
            if chit_detected:
                self.consecutive_contour_frames += 1
                if self.consecutive_contour_frames >= 3:
                    # Object validation check: continue only if paper confidence is high
                    if paper_conf > 0.55:
                        self.state = "CHEAT_SHEET_HELD"
                        self.score_card["follows_hand"] = 3
                        self.cheat_sheet_lost_frames = 0
                        reason = f"Object validated as paper-like (paper confidence: {paper_conf:.2f})."
                    else:
                        self.state = "NORMAL"
                        reason = f"Object rejected by validator (paper confidence: {paper_conf:.2f}, bottle: {bottle_conf:.2f})."
            else:
                self.state = "NORMAL"
                reason = "Object disappeared immediately."
                
        elif self.state == "CHEAT_SHEET_HELD":
            if chit_detected:
                self.cheat_sheet_lost_frames = 0
                recent_down = self.gaze_history[-10:].count("DOWN") >= 2
                if gaze == "DOWN" or recent_down:
                    self.state = "READING_CHEAT_SHEET"
                    self.consecutive_down_gaze = 1
                    self.score_card["down_gaze"] = 3
                    reason = "Student repeatedly looking DOWN at the object."
            else:
                self.cheat_sheet_lost_frames += 1
                if self.cheat_sheet_lost_frames >= 20:
                    self.state = "NORMAL"
                    reason = "Cheat sheet lost for 20 frames."
                    
        elif self.state == "READING_CHEAT_SHEET":
            if chit_detected:
                self.cheat_sheet_lost_frames = 0
                if gaze == "DOWN" or gaze == "LEFT":
                    self.consecutive_down_gaze += 1
            else:
                self.cheat_sheet_lost_frames += 1
                if self.cheat_sheet_lost_frames >= 20:
                    recent_down_or_left = self.gaze_history[-10:].count("DOWN") + self.gaze_history[-10:].count("LEFT") >= 3
                    if gaze in ("DOWN", "LEFT", "FORWARD") or recent_down_or_left:
                        self.state = "CHEAT_SHEET_UNDER_ANSWER_SHEET"
                        self.under_sheet_frames = 1
                        self.score_card["hidden_under_paper"] = 4
                        reason = "Object disappeared beneath answer sheet while writing continues."
                    else:
                        self.state = "NORMAL"
                        reason = "Cheat sheet lost and gaze changes."
                    
        elif self.state == "CHEAT_SHEET_UNDER_ANSWER_SHEET":
            self.under_sheet_frames += 1
            if self.return_pocket_motion_confirmed:
                self.state = "RETURN_TO_POCKET"
                self.return_pocket_frames = 1
                reason = "Return to pocket motion confirmed (sheet stowed)."
                self.return_pocket_motion_confirmed = False
            elif self.under_sheet_frames > 350:  # Timeout
                self.state = "NORMAL"
                
        elif self.state == "RETURN_TO_POCKET":
            self.return_pocket_frames += 1
            if not chit_detected and not hand_near_pocket and self.return_pocket_frames >= 5:
                self.state = "CHEATING_CONFIRMED"
                reason = "Hand left pocket area and returned to normal exam posture."
                
        elif self.state == "CHEATING_CONFIRMED":
            pass # Persistent latch
            
        if self.state != prev_state:
            self.state_history.append((prev_state, self.state))
            
            # Print transition event to console
            print("\n----------------------------------------------------")
            print(f"{timestamp_str}")
            print("State changed:")
            print(f"Student ID {self.student_id}")
            print(f"{prev_state} → {self.state}")
            if reason:
                print(f"Reason: {reason}")
            print("----------------------------------------------------")
            
            # If cheating is confirmed, print the final summary block
            if self.state == "CHEATING_CONFIRMED":
                print("\n====================================================")
                print("🚨 CHEATING DETECTED")
                print(f"Student ID : {self.student_id}")
                print(f"Trajectory : {' → '.join(self.visited_regions)}")
                print(f"Evidence Score : {self.get_evidence_score()}")
                print("\nEvidence Summary")
                print("✔ Pocket reach detected")
                print("✔ New object appeared")
                print("✔ Object followed student's hand")
                print("✔ Student looked at object")
                print("✔ Object hidden under answer sheet")
                print("✔ Object returned to pocket")
                print("✔ Complete temporal sequence verified")
                print("\nReason:")
                print("Unauthorized cheat sheet used during examination.")
                print("====================================================\n")
            
        return self.state


class StudentAssociationEngine:
    """
    Decoupled association checks utilizing scale-invariant proximity limits.
    """
    @staticmethod
    def associate(sheet_center, wrists, pw, ph):
        """
        wrists: ((lw_x, lw_y), (rw_x, rw_y)) frame absolute coordinates.
        pw, ph: Bounding box dimensions of student.
        """
        diag = np.sqrt(pw**2 + ph**2)
        thresh = diag * 0.15  # 15% of body diagonal
        
        lw, rw = wrists
        cx, cy = sheet_center
        
        if lw is not None:
            if np.sqrt((cx - lw[0])**2 + (cy - lw[1])**2) < thresh:
                return True
        if rw is not None:
            if np.sqrt((cx - rw[0])**2 + (cy - rw[1])**2) < thresh:
                return True
                
        return False


class CheatingDecisionEngine:
    """
    Decides if a student is cheating based on FSM status AND evidence scorecard totals.
    """
    def __init__(self, threshold=12):
        self.threshold = threshold

    def evaluate(self, state_tracker: StudentStateMachine) -> tuple:
        score = state_tracker.get_evidence_score()
        # Verify trajectory does not contain MOUTH
        if "MOUTH" in state_tracker.visited_regions:
            return False, "Trajectory inconsistent with cheat sheet usage (drinking/eating)."
            
        if state_tracker.state == "CHEATING_CONFIRMED" and score >= self.threshold:
            return True, "Unauthorized cheat sheet used during examination."
        return False, ""
