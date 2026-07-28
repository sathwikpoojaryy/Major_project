import numpy as np
import json
import os

class ObjectOwnershipTracker:
    """
    Tracks cheat sheet ownership across frames.
    Prevents ownership flickering by requiring stable association of 3-5 frames.
    """
    def __init__(self):
        self.sheet_db = {}
        self.sheet_id_counter = 0

    def update(self, validated_sheets, active_students, fidx):
        """
        validated_sheets: list of dict, each:
          { "bbox": (x1,y1,x2,y2), "center": (cx,cy), "confidence": float, "detected_by_tid": int }
        active_students: dict of student positions
        
        Returns:
            list of tuple: list of (sheet_id, old_owner, new_owner) events
        """
        events = []
        current_sheet_ids = []

        for sheet in validated_sheets:
            cx, cy = sheet["center"]
            curr_detect_tid = sheet["detected_by_tid"]

            # Match sheet to existing database by spatial distance (< 60 pixels)
            matched_id = None
            min_d = 60
            for sid, sinfo in self.sheet_db.items():
                prev_cx, prev_cy = sinfo["last_center"]
                d = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                if d < min_d:
                    min_d = d
                    matched_id = sid

            if matched_id is not None:
                # Update existing sheet record
                sinfo = self.sheet_db[matched_id]
                sinfo["last_center"] = (cx, cy)
                sinfo["last_frame"] = fidx

                # Check if detector ID changed
                if str(curr_detect_tid) != str(sinfo["current_owner"]):
                    if str(curr_detect_tid) == str(sinfo["candidate_owner"]):
                        sinfo["candidate_frames"] += 1
                        # 4 consecutive frames (0.2s at 20fps) of stable association to transfer
                        if sinfo["candidate_frames"] >= 4:
                            old_owner = sinfo["current_owner"]
                            sinfo["current_owner"] = curr_detect_tid
                            sinfo["candidate_owner"] = None
                            sinfo["candidate_frames"] = 0
                            sinfo["stable_frames"] = 1
                            events.append((matched_id, old_owner, curr_detect_tid))
                    else:
                        sinfo["candidate_owner"] = curr_detect_tid
                        sinfo["candidate_frames"] = 1
                else:
                    sinfo["candidate_owner"] = None
                    sinfo["candidate_frames"] = 0
                    sinfo["stable_frames"] += 1
                
                current_sheet_ids.append(matched_id)
            else:
                # New sheet found
                self.sheet_id_counter += 1
                new_id = self.sheet_id_counter
                self.sheet_db[new_id] = {
                    "last_center": (cx, cy),
                    "last_frame": fidx,
                    "current_owner": curr_detect_tid,
                    "candidate_owner": None,
                    "candidate_frames": 0,
                    "stable_frames": 1
                }
                current_sheet_ids.append(new_id)

        # Cleanup stale tracks (lost for > 30 frames)
        for sid in list(self.sheet_db.keys()):
            if fidx - self.sheet_db[sid]["last_frame"] > 30:
                del self.sheet_db[sid]

        return events


class PassingStateMachine:
    """
    Symmetric state machine for a pair of students tracking cheat sheet passing:
    NO_INTERACTION -> HAND_APPROACH -> OBJECT_IN_TRANSFER -> OWNERSHIP_TRANSFER -> OBJECT_RECEIVED -> PASSING_CONFIRMED
    """
    def __init__(self, student_A, student_B):
        self.student_A = min(student_A, student_B)
        self.student_B = max(student_A, student_B)
        self.state = "NO_INTERACTION"
        self.state_history = []
        self.initial_holder = None
        
        self.score_card = {
            "hands_approached": 0,
            "object_in_transfer": 0,
            "ownership_transferred": 0,
            "stable_ownership": 0,
            "students_separated": 0
        }

    def get_score(self):
        return sum(self.score_card.values())

    def update(self, hands_close: bool, obj_in_middle: bool, ownership_event: bool, 
               stable_possession: bool, separated: bool, timestamp_str: str):
        prev_state = self.state
        reason = ""

        # State machine FSM transitions
        if self.state == "NO_INTERACTION":
            if hands_close:
                self.state = "HAND_APPROACH"
                self.score_card["hands_approached"] = 3
                reason = "Students' wrists moved close together."

        elif self.state == "HAND_APPROACH":
            if obj_in_middle:
                self.state = "OBJECT_IN_TRANSFER"
                self.score_card["object_in_transfer"] = 4
                reason = "Cheat sheet detected in the midpoint region between their wrists."
            elif not hands_close:
                self.state = "NO_INTERACTION"
                reason = "Students separated hands without passing anything."

        elif self.state == "OBJECT_IN_TRANSFER":
            if ownership_event:
                self.state = "OWNERSHIP_TRANSFER"
                self.score_card["ownership_transferred"] = 5
                reason = f"Cheat sheet ownership transferred from Student {self.student_A} to Student {self.student_B}."
            elif not hands_close:
                self.state = "NO_INTERACTION"
                reason = "Hands separated before sheet transfer completed."

        elif self.state == "OWNERSHIP_TRANSFER":
            if stable_possession:
                self.state = "OBJECT_RECEIVED"
                self.score_card["stable_ownership"] = 2
                reason = "New owner maintained stable possession of the cheat sheet."
            elif not hands_close:
                self.state = "NO_INTERACTION"
                reason = "Hands separated before ownership stabilized."

        elif self.state == "OBJECT_RECEIVED":
            if separated:
                self.state = "PASSING_CONFIRMED"
                self.score_card["students_separated"] = 3
                reason = "Students separated after sheet transfer."
            elif not hands_close:
                # If they keep hands close, wait, but if they separate, it's confirmed
                pass

        elif self.state == "PASSING_CONFIRMED":
            pass # Persistent latch

        if self.state != prev_state:
            self.state_history.append((prev_state, self.state))
            
            # Print transition event to console
            print("\n----------------------------------------------------")
            print(f"{timestamp_str}")
            print("State changed:")
            print(f"Student Pair ({self.student_A}, {self.student_B})")
            print(f"{prev_state} → {self.state}")
            if reason:
                print(f"Reason: {reason}")
            print("----------------------------------------------------")

            if self.state == "PASSING_CONFIRMED":
                print("\n====================================================")
                print("🚨 CHEAT SHEET PASSING DETECTED")
                print(f"Student A : {self.student_A}")
                print(f"Student B : {self.student_B}")
                print("\nEvidence Summary")
                print("✔ Hands approached")
                print("✔ Cheat sheet shared")
                print("✔ Ownership transferred")
                print("✔ Student B received object")
                print("✔ Temporal sequence verified")
                print("\nReason:")
                print("Unauthorized transfer of cheat sheet between students.")
                print("====================================================\n")

        return self.state


class PassingDecisionEngine:
    def __init__(self, threshold=12):
        self.threshold = threshold

    def evaluate(self, state_tracker: PassingStateMachine) -> bool:
        score = state_tracker.get_score()
        return state_tracker.state == "PASSING_CONFIRMED" and score >= self.threshold


class InteractionDetector:
    """
    Main orchestrator that performs hand proximity checks, updates ownership, 
    drives state machines, and runs the decision engine.
    """
    def __init__(self):
        self.ownership_tracker = ObjectOwnershipTracker()
        self.state_machines = {}
        self.decision_engine = PassingDecisionEngine()
        self.confirmed_passing_pairs = set()
        self.student_sheet_frames = {}
        self.student_sheet_history = {}

    def update(self, active_students, validated_sheets, timestamp_str, fidx):
        """
        active_students: dict of student positions
        validated_sheets: list of validated cheat sheets
        timestamp_str: mm:ss.ff video timestamp
        fidx: current frame index
        """
        # 1. Update consecutive sheet frames and sheet history for all active students
        sheet_owners_this_frame = set(str(sheet["detected_by_tid"]) for sheet in validated_sheets)
        for sheet in validated_sheets:
            self.student_sheet_history[str(sheet["detected_by_tid"])] = fidx

        for tid in active_students.keys():
            if str(tid) in sheet_owners_this_frame:
                self.student_sheet_frames[str(tid)] = self.student_sheet_frames.get(str(tid), 0) + 1
            else:
                self.student_sheet_frames[str(tid)] = 0

        # 2. Iterate through every pair of active students to evaluate interaction
        student_ids = list(active_students.keys())
        for i in range(len(student_ids)):
            for j in range(i + 1, len(student_ids)):
                tid_1 = student_ids[i]
                tid_2 = student_ids[j]
                
                # Consistent sorting for keys
                if str(tid_1) < str(tid_2):
                    tid_A, tid_B = tid_1, tid_2
                else:
                    tid_A, tid_B = tid_2, tid_1
                
                pair_key = (tid_A, tid_B)
                if pair_key not in self.state_machines:
                    self.state_machines[pair_key] = PassingStateMachine(tid_A, tid_B)
                fsm = self.state_machines[pair_key]

                # Get student positions
                sA = active_students[tid_A]
                sB = active_students[tid_B]

                lw_A, rw_A = sA["wrists"]
                lw_B, rw_B = sB["wrists"]

                # We require at least one wrist to be tracked for both students
                if (lw_A is None and rw_A is None) or (lw_B is None and rw_B is None):
                    continue

                # Find the closest hands between both students
                wrists_A = [w for w in (lw_A, rw_A) if w is not None]
                wrists_B = [w for w in (lw_B, rw_B) if w is not None]

                min_hand_dist = 9999
                best_wA, best_wB = None, None
                for wA in wrists_A:
                    for wB in wrists_B:
                        dist = np.sqrt((wA[0] - wB[0])**2 + (wA[1] - wB[1])**2)
                        if dist < min_hand_dist:
                            min_hand_dist = dist
                            best_wA, best_wB = wA, wB

                # Scale-invariant hand proximity threshold (25% of average diagonal)
                diag_A = np.sqrt(sA["pw"]**2 + sA["ph"]**2)
                diag_B = np.sqrt(sB["pw"]**2 + sB["ph"]**2)
                avg_diag = (diag_A + diag_B) / 2.0
                hand_close_thresh = avg_diag * 0.25

                hands_close = min_hand_dist < hand_close_thresh
                separated = min_hand_dist > hand_close_thresh * 1.5

                # 3. Check if sheet is held/detected by either student A or B
                obj_in_middle = False
                if hands_close:
                    for sheet in validated_sheets:
                        if str(sheet["detected_by_tid"]) in (str(tid_A), str(tid_B)):
                            obj_in_middle = True
                            break

                # 4. Check for ownership transfer symmetrically
                has_transfer = False
                stable_possession = False
                if hands_close:
                    # Case 1: tid_A had it recently, tid_B has it now
                    had_A = (fidx - self.student_sheet_history.get(str(tid_A), -999) < 60)
                    frames_B = self.student_sheet_frames.get(str(tid_B), 0)
                    if had_A:
                        if frames_B >= 3:
                            has_transfer = True
                        if frames_B >= 5:
                            stable_possession = True

                    # Case 2: tid_B had it recently, tid_A has it now
                    had_B = (fidx - self.student_sheet_history.get(str(tid_B), -999) < 60)
                    frames_A = self.student_sheet_frames.get(str(tid_A), 0)
                    if had_B:
                        if frames_A >= 3:
                            has_transfer = True
                        if frames_A >= 5:
                            stable_possession = True

                # 5. Update pair state machine
                fsm.update(hands_close, obj_in_middle, has_transfer, stable_possession, separated, timestamp_str)

                # 6. Evaluate cheating decision
                if self.decision_engine.evaluate(fsm):
                    self.confirmed_passing_pairs.add(pair_key)

        return self.confirmed_passing_pairs
