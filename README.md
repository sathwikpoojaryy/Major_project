# Major_project
# 🛡️ ExamGuard

### Intelligent Surveillance System for Cheating Detection in Offline Exams

The core idea is:

> **Use computer vision to monitor students during an offline examination and detect suspicious behavior by combining person tracking, head-pose estimation, object detection, hand movement, temporal behavior analysis, and student-to-student interaction analysis.**
> 

---

# 1. First understand the project in ONE picture

Your project can be understood as this:

```
                         INPUT
                           │
                           ▼
                  ┌─────────────────┐
                  │   Exam Video    │
                  │ input_exam.mp4  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     YOLOv8      │
                  │ Detect Students │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    DeepSORT     │
                  │ Assign Student  │
                  │      IDs        │
                  └────────┬────────┘
                           │
             ┌─────────────┼──────────────┐
             │             │              │
             ▼             ▼              ▼
       MediaPipe       MediaPipe      White Object
         Pose          Face Mesh        Detection
             │             │              │
             ▼             ▼              ▼
       Wrist / Hip     Face Landmarks   White Objects
             │             │              │
             │             ▼              ▼
             │          solvePnP     ObjectValidator
             │             │              │
             │             ▼              │
             │       Head Direction       │
             │                            │
             └────────────┬───────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Student State       │
                │ Machine             │
                │                     │
                │ NORMAL              │
                │ POCKET_REACH        │
                │ OBJECT_APPEARS      │
                │ SHEET_HELD          │
                │ READING             │
                │ UNDER_ANSWER_SHEET  │
                │ RETURN_TO_POCKET    │
                │ CONFIRMED           │
                └──────────┬──────────┘
                           │
                           ▼
                 Cheating Decision
                      Engine
                           │
                           ▼
                  Individual Student
                       Decision
                           │
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
   Student-to-Student                 Visualization
     Interaction                         / Output
          │                                 │
          ▼                                 ▼
   Passing Detection                  Bounding Boxes
          │                            Alerts / Labels
          ▼                            Output Video
   PASSING CONFIRMED
```

And before all of this:

```
              NORMAL EXAM VIDEO
                     │
                     ▼
              BaselineLearner
                     │
                     ▼
                baseline.json
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Static papers          Normal movement
    / desk objects          / gaze / wrist
```

That is the **complete high-level architecture**.

---

# 2. What problem are you solving?

Traditional offline exams have invigilators.

An invigilator has to manually observe:

- students looking around
- students taking something from their pocket
- students using hidden notes
- students passing papers
- suspicious hand movements
- unusual interactions

The problem is:

> One human cannot continuously monitor every student with the same attention.
> 

So your project attempts to provide an **automated visual surveillance assistant**.

It doesn't simply say:

> "Student looked left → cheating."
> 

Instead, your project tries to understand a **sequence of actions**.

For example:

```
Hand moves toward pocket
        ↓
Object appears
        ↓
Object resembles paper
        ↓
Paper follows hand
        ↓
Student looks down
        ↓
Paper disappears under answer sheet
        ↓
Hand returns toward pocket
        ↓
Suspicious sequence completed
        ↓
CHEATING CONFIRMED
```

This temporal sequence is one of the strongest ideas in your project.

---

# 3. Your three main objectives

You originally defined three objectives.

## Objective 1 — Student detection and tracking

> Detect and track students by assigning unique IDs using YOLOv8 and DeepSORT.
> 

Meaning:

```
YOLOv8
   ↓
Find people

DeepSORT
   ↓
Maintain identity
```

Example:

```
Frame 1 → Student ID 7
Frame 2 → Student ID 7
Frame 3 → Student ID 7
...
```

Even though the student's position changes, the system attempts to keep the same ID.

---

# 4. Objective 2 — Head direction analysis

You want to determine whether a student is:

```
LEFT
RIGHT
UP
DOWN
FORWARD / CENTER
BACK
```

You use:

```
MediaPipe Face Mesh
        ↓
Facial landmarks
        ↓
solvePnP
        ↓
Head rotation
        ↓
Yaw / Pitch
        ↓
Head direction
```

For example:

```
Yaw = -30°
        ↓
LEFT

Yaw = +30°
        ↓
RIGHT

Pitch = -25°
        ↓
DOWN
```

Your project then uses this information as **evidence**, not as the sole reason for declaring cheating.

---

# 5. Objective 3 — Temporal behavior analysis

This is probably the most important concept to understand for your viva.

A single frame isn't enough.

Consider:

```
Frame 100:
Student looks down
```

Is that cheating?

No.

They may be writing.

But:

```
Frame 100:
Normal

Frame 120:
Hand moves toward pocket

Frame 130:
Object appears

Frame 140:
Object follows hand

Frame 150:
Student looks down

Frame 170:
Object disappears under answer sheet

Frame 200:
Hand returns toward pocket
```

Now there is a **sequence**.

That's why you have:

# `StudentStateMachine`

It remembers what happened previously.

---

# 6. The complete system has TWO phases

This is extremely important.

Your project is not simply:

```
Video → cheating detection
```

It is:

```
PHASE 1
Normal video
     ↓
Baseline learning
     ↓
baseline.json

PHASE 2
Exam video
     ↓
Detection
     ↓
Tracking
     ↓
Behavior analysis
     ↓
Decision
```

Let's understand both.

---

# PART A — BASELINE LEARNING

# 7. What is baseline learning?

Before detecting abnormal behavior, your system needs to know what normal looks like.

Suppose the answer sheet is always on the desk.

The system shouldn't repeatedly detect it as a suspicious white paper.

So you give it a normal exam video:

```
input_normal.mp4
```

Then:

```
BaselineLearner
```

analyzes it.

Output:

```
baseline.json
```

---

# 8. What does `baseline.json` contain?

Three major categories.

### A. Static desk objects

Example:

```
static_paper_0
```

with:

```
bbox
area
aspect_ratio
```

Meaning:

> This white object is normally present here.
> 

---

### B. Normal writing posture

Example:

```
pitch_mean
pitch_std
yaw_mean
yaw_std
```

Meaning:

> This is approximately how students normally orient their heads.
> 

---

### C. Normal wrist movement

Example:

```
min_wrist_to_hip_ratio
pocket_proximity_thresh_ratio
```

Meaning:

> This is the normal range of hand movement relative to the body.
> 

---

# 9. Why normalize wrist distance?

Suppose Student A is:

```
500 px tall
```

and Student B:

```
800 px tall
```

A raw distance isn't a fair comparison.

Instead:

```
wrist-to-hip distance
        ÷
student height
```

gives a ratio.

Example:

```
Student A:
50 / 500 = 0.10

Student B:
80 / 800 = 0.10
```

Both have the same relative movement.

That's why your baseline uses ratios.

---

# 10. How the baseline learns static papers

It looks for white/off-white contours using HSV.

```
BGR image
   ↓
HSV
   ↓
White mask
   ↓
Contours
   ↓
Candidate papers
```

Then candidates close to each other across frames are merged.

If something appears consistently:

```
Frame 1
Frame 2
Frame 3
...
Frame 200
```

it becomes a:

```
static_desk_object
```

Then the actual detection system can ignore it.

---

# PART B — ACTUAL EXAM DETECTION

Now suppose you have:

```
input_exam.mp4
```

The system processes it frame-by-frame.

---

# 11. Step 1 — Read a frame

OpenCV:

```
cap.read()
```

gets:

```
Frame 1
```

Then:

```
Frame 2
```

and so on.

---

# 12. Step 2 — YOLOv8 detects students

YOLO looks at the frame.

Example:

```
             Camera Frame

 ┌──────────────────────────────┐
 │                              │
 │   ┌───────┐      ┌───────┐ │
 │   │Person │      │Person │ │
 │   └───────┘      └───────┘ │
 │                              │
 │          ┌───────┐           │
 │          │Person │           │
 │          └───────┘           │
 └──────────────────────────────┘
```

YOLO provides bounding boxes.

---

# 13. Why YOLO confidence thresholds?

You have different thresholds such as:

```
Detection confidence
Tracking confidence
```

The idea is:

> Don't accept extremely uncertain detections.
> 

For example:

```
YOLO confidence = 0.92
```

Good.

```
YOLO confidence = 0.20
```

Probably unreliable.

So low-confidence detections are filtered.

---

# 14. Step 3 — DeepSORT assigns IDs

YOLO only tells you:

```
Person here
Person there
Person there
```

It doesn't inherently mean:

```
That's the same person as in the previous frame.
```

DeepSORT solves that.

Example:

```
Frame 1:

Student → ID 7

Frame 2:

Student moved

        ↓

Still → ID 7

Frame 3:

Student moved again

        ↓

Still → ID 7
```

This is essential because all later behavior information must belong to a particular student.

---

# 15. Why unique IDs are important

Imagine:

```
Student 7 → looks down
Student 12 → reaches pocket
Student 20 → holds paper
```

You need to know **which behavior belongs to which student**.

Without tracking:

```
Somebody looked down
Somebody moved hand
Somebody had paper
```

You can't connect those events.

With IDs:

```
ID 7:
    looking down

ID 12:
    pocket reach

ID 20:
    paper detected
```

Now the system can build an individual behavior history.

---

# 16. Step 4 — Crop each student

For every tracked student:

```
person_crop=frame[ty1:ty2,tx1:tx2]
```

Now the system isolates:

```
Student 7
```

instead of processing the entire classroom for every operation.

---

# 17. Step 5 — MediaPipe Pose

Pose estimation finds body landmarks.

Important landmarks for your project:

```
Left wrist
Right wrist

Left hip
Right hip
```

Conceptually:

```
          Head
           ●
          / \
         /   \
        ●     ●
      shoulder shoulder
       |       |
       |       |
      ●         ●
    wrist      wrist
       \       /
        \     /
         ●   ●
        hip hip
```

You mainly care about:

```
wrists
hips
```

---

# 18. Why are wrists important?

Because cheating-related actions may involve hand movement.

For example:

```
Normal writing:

        hand
          ↓
       answer sheet
```

versus:

```
Pocket reach:

        hand
          ↓
       pocket
```

Your system measures wrist-to-hip relationships.

---

# 19. Pocket reach

Suppose the wrist becomes very close to the hip.

The system checks something like:

```
wrist-hip distance
        ↓
normalize by student height
        ↓
compare with baseline threshold
```

If sufficiently close:

```
Possible pocket reach
```

But again:

> A single close wrist isn't automatically cheating.
> 

It contributes to the temporal sequence.

---

# 20. Step 6 — Face Mesh

Now your system analyzes the student's head.

MediaPipe Face Mesh provides many facial landmarks.

Your code selects six important points:

```
[1,152,33,263,61,291]
```

These points are used for head-pose estimation.

---

# 21. Step 7 — `solvePnP`

This is an important viva topic.

`solvePnP` means:

> Solve Perspective-n-Point.
> 

You have:

```
3D reference face points
        +
2D detected facial points
        ↓
estimate head rotation
```

Then:

```
Rotation
   ↓
Yaw
Pitch
Roll
```

Your project mainly uses:

```
Yaw
Pitch
```

---

# 22. What does yaw mean?

Think:

```
        LEFT        RIGHT
          ←          →

             🙂
```

Yaw describes left-right rotation.

---

# 23. What does pitch mean?

Think:

```
             UP
             ↑

             🙂

             ↓
            DOWN
```

Pitch describes up-down movement.

For an exam:

```
Looking at answer sheet
        ↓
DOWN
```

is usually normal.

But:

```
Repeatedly looking toward another student
```

could become suspicious when combined with other evidence.

---

# 24. Head direction classification

Your system converts numerical angles into categories.

Conceptually:

```
Yaw < -20°
    ↓
LEFT

Yaw > +20°
    ↓
RIGHT

Pitch < -18°
    ↓
DOWN

Pitch > +18°
    ↓
UP
```

You also have logic for:

```
BACK
```

with confirmation over multiple frames.

This prevents tiny angle fluctuations from constantly changing the direction.

---

# 25. Why smoothing?

Computer vision isn't perfectly stable.

One frame might say:

```
DOWN
```

next:

```
CENTER
```

next:

```
DOWN
```

even though the student barely moved.

So you maintain a smoothing buffer.

For example:

```
Recent 10 frames:

DOWN
DOWN
DOWN
DOWN
DOWN
DOWN
...
```

Now you can confidently say:

```
Student is looking down.
```

This reduces noise.

---

# 26. Step 8 — Detect white objects

Now comes:

# `CheatSheetDetector`

This is responsible for finding suspicious white objects.

Why white?

Because a cheat sheet is likely to be:

```
white/off-white paper
```

Your detector converts the crop:

```
BGR
 ↓
HSV
 ↓
White mask
 ↓
Contours
```

---

# 27. Important distinction

`CheatSheetDetector` does **not** mean:

> "I found cheating."
> 

It means:

> "I found a white object that may be relevant."
> 

That's a very important distinction for your viva.

---

# 28. What happens to each contour?

For every contour, it calculates:

```
x
y
width
height
area
```

Then it filters out tiny/unreasonable objects.

For example:

```
Tiny white noise
       ↓
   discard
```

while:

```
Reasonable white object
       ↓
candidate
```

---

# 29. `ObjectValidator`

Now:

```
CheatSheetDetector
        ↓
ObjectValidator
```

The validator asks:

> What does this object look like geometrically?
> 

It calculates:

### Aspect ratio

```
width / height
```

and:

### Extent

```
contour area
----------------
bounding-box area
```

---

# 30. Why distinguish paper and bottle?

Because white objects can be many things.

For example:

```
Paper
Bottle
White desk object
Noise
```

You don't want:

```
White bottle
    ↓
CHEAT SHEET
```

So `ObjectValidator` gives heuristic scores:

```
paper confidence
bottle confidence
unknown confidence
```

Important:

> These are **heuristic scores**, not probabilities produced by a trained classifier.
> 

---

# 31. Example

Suppose:

```
width = 100
height = 150
```

Aspect ratio:

```
100 / 150
= 0.67
```

Suppose the contour fills much of its bounding box.

That may look paper-like.

Another object:

```
width = 30
height = 100
```

Aspect ratio:

```
0.30
```

Very tall and narrow.

That can look more bottle-like.

---

# 32. Baseline removes normal papers

Suppose the normal answer sheet is already on the desk.

Baseline learned:

```
static_paper_0
```

During actual detection:

```
White object detected
        ↓
Compare with baseline
        ↓
Matches normal static paper?
        ↓
YES
        ↓
Ignore
```

This is how baseline reduces false positives.

---

# 33. Wrist proximity

The detector also checks:

> Is this white object close to the student's hand?
> 

For example:

```
Student
   👤
  /|\
 / | \
✋  📄
```

If paper is far away:

```
Student       📄
```

it is less likely to be something the student is holding.

If:

```
✋📄
```

then it becomes more relevant.

---

# 34. Why scale-invariant thresholds?

Your code uses the student's body dimensions to calculate thresholds.

For example:

```
threshold = percentage of body diagonal
```

Instead of:

```
distance < 50 pixels
```

Why?

Because students can appear at different distances from the camera.

Near student:

```
large bounding box
```

Far student:

```
small bounding box
```

Relative measurements are more robust.

---

# 35. Step 9 — Student State Machine

Now all the information comes together.

This is the heart of the behavior logic.

The state machine tracks:

```
What is happening now?
+
What happened before?
```

---

# 36. The states

Your project has approximately:

```
NORMAL
   ↓
POCKET_REACH
   ↓
UNKNOWN_OBJECT_APPEARS
   ↓
CHEAT_SHEET_HELD
   ↓
READING_CHEAT_SHEET
   ↓
CHEAT_SHEET_UNDER_ANSWER_SHEET
   ↓
RETURN_TO_POCKET
   ↓
CHEATING_CONFIRMED
```

This is much more powerful than checking one frame.

---

# 37. State 1 — NORMAL

Initially:

```
Student = NORMAL
```

Student is:

```
writing
looking normally
moving hands normally
```

Nothing suspicious.

---

# 38. State 2 — POCKET_REACH

Suppose:

```
wrist gets close to hip/pocket
```

for sufficient frames.

Then:

```
NORMAL
   ↓
POCKET_REACH
```

Your counters prevent a one-frame accidental movement from triggering it.

---

# 39. Why counters?

Suppose:

```
Frame 100 → hand near pocket
Frame 101 → hand normal
```

If you immediately trigger:

```
CHEATING
```

that would be terrible.

Instead:

```
Need multiple frames
```

to establish that a movement is real.

This is called **temporal consistency**.

---

# 40. State 3 — UNKNOWN_OBJECT_APPEARS

Now suppose after reaching the pocket:

```
white object appears
```

Then:

```
POCKET_REACH
       ↓
UNKNOWN_OBJECT_APPEARS
```

At this point the system is suspicious, but not yet sure it's a cheat sheet.

---

# 41. State 4 — CHEAT_SHEET_HELD

If the object:

- persists for several frames
- is paper-like
- is associated with the student's hand

then:

```
UNKNOWN_OBJECT_APPEARS
       ↓
CHEAT_SHEET_HELD
```

Your system requires multiple frames.

This reduces false positives.

---

# 42. Evidence scoring

This is another major concept.

Your system doesn't only use states.

It also accumulates evidence.

For example:

```
Pocket movement          +3
Paper geometry           +4
Object follows hand      +3
Looking down             +3
Paper hidden             +4
Return to pocket         +3
```

The exact values come from your implementation.

So you can think:

```
Suspicious behavior
       ↓
Evidence score increases
```

while some behavior can reduce the score.

For example, your code penalizes a likely bottle:

```
Bottle evidence
    ↓
negative score
```

---

# 43. State 5 — READING_CHEAT_SHEET

Suppose the student is holding a paper-like object and looking down.

The state becomes:

```
CHEAT_SHEET_HELD
       ↓
READING_CHEAT_SHEET
```

Your code uses head direction such as:

```
DOWN
```

and temporal conditions.

Again:

> Looking down alone isn't cheating.
> 

It's suspicious only in combination with the object/behavior context.

---

# 44. State 6 — UNDER ANSWER SHEET

Suppose:

```
Paper-like object disappears
```

but recent behavior indicates the student was reading/writing.

Your system can transition toward:

```
CHEAT_SHEET_UNDER_ANSWER_SHEET
```

The interpretation is:

> The suspicious object may have been moved underneath the answer sheet.
> 

---

# 45. State 7 — RETURN_TO_POCKET

Suppose the student's hand goes back toward the pocket.

Then:

```
CHEAT_SHEET_UNDER_ANSWER_SHEET
             ↓
       RETURN_TO_POCKET
```

---

# 46. State 8 — CHEATING_CONFIRMED

Finally, after the required conditions are satisfied:

```
RETURN_TO_POCKET
       ↓
normal posture for several frames
       ↓
CHEATING_CONFIRMED
```

And your decision engine also requires sufficient evidence.

---

# 47. Why use a state machine instead of an ML model?

This is a very good viva question.

Your answer:

> **“We use a rule-based temporal state machine because cheating is not defined by a single visual feature. It is a sequence of events. The state machine makes the sequence explicit and allows us to combine multiple evidence sources over time.”**
> 

For your project phase, this is a reasonable engineering approach.

---

# 48. `CheatingDecisionEngine`

Now another important distinction.

### State Machine

Answers:

> **What behavioral stage is the student currently in?**
> 

### Decision Engine

Answers:

> **Is there enough evidence to finally declare cheating?**
> 

Your decision engine has a threshold:

```
threshold = 12
```

Conceptually:

```
State = CHEATING_CONFIRMED
        +
Score >= 12
        ↓
CHEATING = TRUE
```

If score is:

```
10
```

then:

```
Not enough evidence
```

---

# 49. Special exclusion — mouth

Your code has an important safety condition:

```
If trajectory contains MOUTH
        ↓
return false
```

Why?

Suppose the student takes a bottle or something toward the mouth.

You don't want the system to interpret that as cheat-sheet behavior.

So:

```
Object → mouth
       ↓
likely normal object use
       ↓
reject cheating decision
```

This is a useful false-positive reduction.

---

# 50. Individual cheating vs passing

Now your project has another layer.

Suppose:

```
Student A → passes paper → Student B
```

This isn't necessarily captured by the individual state machine.

So you have:

# `InteractionDetector`

It analyzes **student-to-student interaction**.

---

# 51. Interaction architecture

```
Student A wrists
       +
Student B wrists
       +
Detected sheets
       ↓
InteractionDetector
       ↓
PassingStateMachine
       ↓
PassingDecisionEngine
       ↓
PASSING_CONFIRMED
```

---

# 52. Detecting close hands

Suppose:

```
Student A       Student B

   ✋              ✋
     \            /
       \        /
         close
```

The system calculates the minimum distance between:

```
A-left wrist
A-right wrist
B-left wrist
B-right wrist
```

It checks all relevant wrist pairs.

---

# 53. Why all wrist combinations?

Because you don't know which hand is being used.

It could be:

```
A-left → B-left
A-left → B-right
A-right → B-left
A-right → B-right
```

So the system checks the minimum distance.

---

# 54. Scale-invariant interaction threshold

Again, it uses body size.

Conceptually:

```
average student body diagonal
        ×
interaction percentage
        =
hand proximity threshold
```

This is better than a fixed pixel threshold.

---

# 55. Passing state machine

Your passing state machine is approximately:

```
NO_INTERACTION
       ↓
HAND_APPROACH
       ↓
OBJECT_IN_TRANSFER
       ↓
OWNERSHIP_TRANSFER
       ↓
OBJECT_RECEIVED
       ↓
PASSING_CONFIRMED
```

---

# 56. Example of paper passing

Imagine:

```
Frame 1

A          B

✋          ✋
far apart
```

State:

```
NO_INTERACTION
```

---

### Frame 20

```
A          B

 ✋        ✋
   close
```

State:

```
HAND_APPROACH
```

Evidence:

```
+3
```

---

### Frame 25

A paper is near the hands:

```
A      📄      B
✋             ✋
```

State:

```
OBJECT_IN_TRANSFER
```

Evidence:

```
+4
```

---

### Frame 30

Paper ownership changes:

```
A → B
```

State:

```
OWNERSHIP_TRANSFER
```

Evidence:

```
+5
```

---

### Frame 35

B has the paper consistently.

```
OBJECT_RECEIVED
```

Evidence:

```
+2
```

---

### Frame 50

Students separate.

```
A                B
✋                ✋
```

State:

```
PASSING_CONFIRMED
```

Evidence:

```
+3
```

Total:

```
3 + 4 + 5 + 2 + 3
= 17
```

Threshold:

```
12
```

Therefore:

```
PASSING CONFIRMED
```

---

# 57. `ObjectOwnershipTracker`

This component is designed to answer:

> Who currently owns the sheet?
> 

Suppose:

```
Frame 1:
Sheet → Student A

Frame 2:
Sheet → Student A

Frame 3:
Sheet → Student B

Frame 4:
Sheet → Student A

Frame 5:
Sheet → Student B
```

If you directly use every frame, ownership would flicker:

```
A → B → A → B
```

That's unreliable.

So the tracker requires multiple consistent frames before changing ownership.

---

# 58. Ownership transfer

Conceptually:

```
Sheet ID 3

Owner:
Student 7
       ↓
candidate owner Student 12
       ↓
Student 12 keeps it for 4 frames
       ↓
ownership transferred
       ↓
Student 12
```

This is temporal stabilization.

---

# 59. Important implementation detail in YOUR current code

This is something you should be careful about during your viva.

Your `InteractionDetector` creates:

```
self.ownership_tracker=ObjectOwnershipTracker()
```

but in the current `update()` code you showed me, it does **not actually call**:

```
self.ownership_tracker.update(...)
```

Instead, the current interaction logic uses:

```
student_sheet_history
+
student_sheet_frames
```

to infer transfer.

So don't tell the evaluator:

> “The ownership tracker is definitely being used for every passing decision.”
> 

The technically accurate explanation is:

> **“ObjectOwnershipTracker is implemented to maintain stable sheet ownership, while the current InteractionDetector also uses frame-based student-sheet history and consecutive-frame counts to infer ownership transfer.”**
> 

That's safer and more accurate.

---

# 60. Complete information flow for ONE student

Let's follow Student ID 7.

```
                    FRAME
                      │
                      ▼
                    YOLO
                      │
                      ▼
                 DeepSORT
                      │
                      ▼
                  Student 7
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
       Pose        Face Mesh    White Mask
          │           │           │
          ▼           ▼           ▼
     Wrists/Hips   Landmarks   Contours
          │           │           │
          │           ▼           ▼
          │        solvePnP   ObjectValidator
          │           │           │
          ▼           ▼           ▼
    Wrist motion   Head dir   Object type
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
              StudentStateMachine
                      │
                      ▼
                Evidence Score
                      │
                      ▼
             CheatingDecisionEngine
                      │
                      ▼
                Student 7:
              CHEATING / NORMAL
```

---

# 61. Complete information flow for TWO students

Now Student 7 and Student 12.

```
Student 7 ──┐
            │
            ├── wrists
            │
            ├── sheet
            │
            ▼
       InteractionDetector
            ▲
            │
            ├── wrists
            │
            ├── sheet
            │
Student 12 ─┘
            │
            ▼
    Hand distance analysis
            │
            ▼
      Sheet ownership
            │
            ▼
    Passing State Machine
            │
            ▼
    Passing Decision Engine
            │
            ▼
     PASSING CONFIRMED
```

---

# 62. What happens in every frame?

This is a good way to memorize the whole project.

For every frame:

```
1. Read frame
       ↓
2. YOLO detects people
       ↓
3. DeepSORT assigns IDs
       ↓
4. For each student:
       ↓
   Crop student
       ↓
5. MediaPipe Pose
       ↓
   wrists + hips
       ↓
6. Calculate hand movement
       ↓
7. Face Mesh
       ↓
8. solvePnP
       ↓
   yaw + pitch
       ↓
9. Classify head direction
       ↓
10. Detect white objects
       ↓
11. Validate object geometry
       ↓
12. Compare with baseline
       ↓
13. Update student's state
       ↓
14. Update evidence score
       ↓
15. Make individual decision
       ↓
16. Check student pairs
       ↓
17. Detect possible passing
       ↓
18. Draw results
       ↓
19. Save output frame
```

Then:

```
Next frame
   ↓
Repeat
```

---

# 63. Why is temporal analysis so important?

Let's compare two systems.

### System A — frame-based

```
Paper detected
    ↓
CHEATING
```

Problem:

```
False positives ↑
```

Because paper could simply be the answer sheet.

---

### Your system

```
Pocket movement
      ↓
Object appears
      ↓
Paper-like
      ↓
Hand association
      ↓
Gaze behavior
      ↓
Object hidden
      ↓
Return movement
      ↓
Enough evidence
      ↓
CHEATING
```

This is much more context-aware.

---

# 64. What is the role of every major technology?

You should be able to answer this instantly.

| Technology | Role |
| --- | --- |
| **Python** | Main programming language |
| **OpenCV** | Video processing, image operations, contours, drawing |
| **YOLOv8** | Detect students |
| **DeepSORT** | Track students and maintain IDs |
| **MediaPipe Pose** | Detect wrists, hips and body landmarks |
| **MediaPipe Face Mesh** | Detect facial landmarks |
| **OpenCV solvePnP** | Estimate head orientation |
| **HSV thresholding** | Detect white/off-white objects |
| **ObjectValidator** | Geometric classification of candidate objects |
| **BaselineLearner** | Learn normal exam environment/behavior |
| **StudentStateMachine** | Track suspicious behavior over time |
| **CheatingDecisionEngine** | Make final individual decision |
| **InteractionDetector** | Detect student-to-student interactions |
| **PassingStateMachine** | Track paper-passing sequence |
| **PassingDecisionEngine** | Confirm passing |
| **JSON** | Store baseline information |
| **OpenCV VideoWriter** | Save processed output video |

---

# 65. What exactly is AI/ML in your project?

This is another likely viva question.

Your project uses a **hybrid approach**.

You are not training one giant neural network to say:

```
CHEATING / NOT CHEATING
```

Instead:

```
YOLOv8
   ↓
Deep learning detection

MediaPipe
   ↓
Pre-trained landmark models

DeepSORT
   ↓
Tracking algorithm

OpenCV
   ↓
Computer vision

Rules + State Machines
   ↓
Temporal behavior reasoning
```

So you can say:

> **“ExamGuard follows a hybrid computer-vision architecture. It uses pretrained deep-learning models for person detection and landmark extraction, while higher-level cheating decisions are made using geometric heuristics, temporal state machines, baseline statistics, and evidence scoring.”**
> 

That is a very good project explanation.

---

# 66. Is YOLO detecting cheating?

**No.**

YOLO detects:

```
PERSON
```

It doesn't know:

```
cheating
```

Your architecture is:

```
YOLO
 ↓
Person detection
```

not:

```
YOLO
 ↓
Cheating detection
```

---

# 67. Is MediaPipe detecting cheating?

**No.**

MediaPipe gives:

```
body landmarks
face landmarks
```

Your logic interprets those landmarks.

---

# 68. Is OpenCV detecting cheating?

Not directly.

OpenCV provides:

```
video processing
HSV segmentation
contours
geometry
solvePnP
drawing
```

Then your application logic decides what those measurements mean.

---

# 69. Where does the actual cheating decision happen?

Mainly:

```
StudentStateMachine
        +
CheatingDecisionEngine
```

And for passing:

```
PassingStateMachine
        +
PassingDecisionEngine
```

---

# 70. Three levels of reasoning

You can explain your project as three layers.

## Layer 1 — Perception

"What do I see?"

```
YOLO
MediaPipe
OpenCV
```

Outputs:

```
student
landmarks
objects
angles
positions
```

---

## Layer 2 — Behavior

"What is the student doing?"

```
State Machine
```

Outputs:

```
pocket reach
object held
reading
hiding
returning
```

---

## Layer 3 — Decision

"Is this cheating?"

```
Decision Engine
```

Outputs:

```
NORMAL
CHEATING
PASSING
```

This is a **very useful way to explain your project architecture**.

---

# 71. Another way to understand the architecture

Think of it like a human invigilator.

### Eyes

```
YOLO
```

Finds students.

### Identity memory

```
DeepSORT
```

Remembers which student is which.

### Body observation

```
MediaPipe Pose
```

Sees hands and body.

### Face observation

```
MediaPipe Face Mesh
+
solvePnP
```

Sees where the student is looking.

### Object observation

```
CheatSheetDetector
+
ObjectValidator
```

Sees suspicious white objects.

### Memory of normal behavior

```
BaselineLearner
```

Knows what normally exists.

### Behavioral memory

```
StudentStateMachine
```

Remembers the sequence.

### Final judgment

```
CheatingDecisionEngine
```

Decides.

### Interaction observer

```
InteractionDetector
```

Checks whether students interact.

That's basically what your system is doing.

---

# 72. Why baseline + state machine together?

This distinction is important.

### Baseline asks:

> What is normal?
> 

For example:

```
Normal wrist movement = 0.25 ratio
Normal paper = this location
Normal gaze = approximately this range
```

### State machine asks:

> What is happening over time?
> 

For example:

```
Pocket
→ paper
→ reading
→ hiding
→ pocket
```

Together:

```
Baseline
   +
Temporal behavior
   ↓
Better cheating detection
```

---

# 73. What happens if a student simply looks left?

```
Student looks LEFT
       ↓
Head direction = LEFT
       ↓
No paper
No pocket reach
No suspicious sequence
       ↓
NORMAL
```

Good.

---

# 74. What if a student looks down?

```
Student looks DOWN
       ↓
Normal during writing
       ↓
No suspicious object
       ↓
NORMAL
```

Good.

---

# 75. What if a white bottle appears?

```
White object
     ↓
ObjectValidator
     ↓
Bottle confidence high
     ↓
Negative evidence
     ↓
Not treated as cheat sheet
```

This helps reduce false positives.

---

# 76. What if the answer sheet is detected?

```
White object
     ↓
Compare with baseline
     ↓
Matches static desk object
     ↓
Ignore
```

Again, this is why baseline is important.

---

# 77. What if a student actually takes a cheat sheet?

Potential sequence:

```
Normal
 ↓
Hand approaches pocket
 ↓
Pocket reach confirmed
 ↓
Object appears
 ↓
Object resembles paper
 ↓
Paper associated with hand
 ↓
Student looks down
 ↓
Paper disappears
 ↓
Return toward pocket
 ↓
Normal posture
 ↓
Enough evidence
 ↓
CHEATING CONFIRMED
```

That's your main use case.

---

# 78. What if Student A gives paper to Student B?

Then:

```
Student A
    +
Student B
    ↓
Hands become close
    ↓
Paper detected
    ↓
A has paper
    ↓
B gets paper
    ↓
B retains paper
    ↓
Students separate
    ↓
PASSING CONFIRMED
```

---

# 79. What does the output look like conceptually?

Your output video can show things like:

```
┌─────────────────────────────────────────────┐
│                                             │
│    ┌──────────────┐                         │
│    │ ID: 7        │                         │
│    │ CHEATING     │ 🚨                     │
│    └──────────────┘                         │
│                                             │
│                  ┌──────────────┐           │
│                  │ ID: 12       │           │
│                  │ NORMAL       │           │
│                  └──────────────┘           │
│                                             │
│       PASSING: 7 ↔ 12                      │
│                                             │
└─────────────────────────────────────────────┘
```

---

# 80. What are the important thresholds?

You should know the concept, not necessarily memorize every number.

Your project contains thresholds for:

### YOLO

```
Detection confidence
Tracking confidence
```

### Head pose

```
Yaw left
Yaw right
Pitch up
Pitch down
```

### Temporal confirmation

```
Back confirmation frames
Object persistence frames
```

### Wrist movement

```
Pocket proximity ratio
```

### Object detection

```
Relative object size
Area
Distance from wrist
```

### Cheating decision

```
Evidence threshold = 12
```

### Passing

```
Evidence threshold = 12
```

---

# 81. Why so many thresholds?

Because computer vision is noisy.

Without thresholds:

```
tiny noise
 ↓
false detection
```

With thresholds:

```
Only sufficiently strong evidence
 ↓
more stable behavior
```

---

# 82. What is the biggest technical challenge?

One of the biggest challenges is:

> **Distinguishing normal exam behavior from suspicious behavior.**
> 

Because these can look similar.

For example:

```
Looking down
```

could mean:

```
writing
```

or:

```
reading a cheat sheet
```

The difference comes from **context**.

That's why you combine:

```
gaze
+
wrist movement
+
object detection
+
object position
+
temporal sequence
```

---

# 83. Another major challenge: occlusion

Suppose a student's hand disappears behind:

```
body
desk
answer sheet
```

Computer vision may temporarily lose it.

That's why your system uses:

```
tracking
+
multiple frames
+
state persistence
```

rather than trusting one frame.

---

# 84. Another challenge: different student sizes

Students can be:

```
near camera
far camera
```

Therefore fixed pixel thresholds are unreliable.

Your project uses relative measurements such as:

```
distance / body height
```

and:

```
distance / body diagonal
```

This improves robustness.

---

# 85. Another challenge: lighting

White paper detection can be affected by:

```
bright light
shadows
reflections
camera exposure
```

That's why your detector uses HSV thresholds and additional geometric filters.

But this is still a limitation.

---

# 86. Limitations of the current project

Be honest about these in your viva.

### 1. White-object assumption

Your cheat-sheet detector is based heavily on:

```
white/off-white
```

So colored or dark cheat sheets may not be detected.

---

### 2. Camera angle

If the camera is poorly positioned:

```
face landmarks
wrists
objects
```

may be difficult to detect.

---

### 3. Occlusion

Students can block:

```
hands
face
paper
```

from the camera.

---

### 4. False positives

Normal movements may sometimes appear suspicious.

For example:

```
Student adjusts clothing
```

could resemble:

```
pocket reach
```

Your temporal rules reduce this, but cannot eliminate it completely.

---

### 5. False negatives

A very small or hidden cheat sheet may not be detected.

---

### 6. Fixed heuristic thresholds

Some thresholds are manually designed:

```
aspect ratio
extent
distance
evidence scores
```

They may need tuning for different classrooms/cameras.

---

# 87. How can the project be improved later?

If they ask:

> "What is future scope?"
> 

You can say:

### 1. Train an object detector

Instead of white-pixel detection:

```
YOLO
 ↓
cheat sheet
bottle
pen
phone
```

---

### 2. Better action recognition

Use:

```
LSTM
Transformer
Temporal CNN
```

to learn suspicious action sequences automatically.

---

### 3. Multi-camera surveillance

```
Camera 1
Camera 2
Camera 3
      ↓
Central monitoring
```

---

### 4. Better face/head-pose estimation

Use more robust 3D head-pose models.

---

### 5. Automatic alert dashboard

```
Student 7
🚨 High risk

Student 12
⚠️ Suspicious

Student 20
✓ Normal
```

---

# 88. What is the role of each file in your project?

Based on the files you've shown me:

```
main.py
    ↓
Main orchestrator
```

```
BaselineLearner
    ↓
Learns normal environment/behavior
```

```
CheatSheetDetector
    ↓
Finds candidate white objects
```

```
ObjectValidator
    ↓
Checks whether candidate resembles paper/bottle
```

```
StudentStateMachine
    ↓
Tracks suspicious behavior sequence
```

```
CheatingDecisionEngine
    ↓
Final individual cheating decision
```

```
StudentAssociationEngine
    ↓
Helper for associating a sheet with a student's wrist
```

```
ObjectOwnershipTracker
    ↓
Tracks sheet ownership over time
```

```
PassingStateMachine
    ↓
Tracks passing sequence
```

```
PassingDecisionEngine
    ↓
Final passing decision
```

```
InteractionDetector
    ↓
Coordinates student-to-student passing detection
```

---

# 89. Your complete project architecture

I would explain it to an evaluator like this:

```
                         EXAM VIDEO
                             │
                             ▼
                     ┌──────────────┐
                     │   OpenCV     │
                     │ Video Input  │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   YOLOv8     │
                     │   Person     │
                     │  Detection   │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   DeepSORT   │
                     │   Tracking   │
                     │  Student ID  │
                     └──────┬───────┘
                            │
              ┌─────────────┼──────────────┐
              │             │              │
              ▼             ▼              ▼
        MediaPipe       Face Mesh      HSV/Object
           Pose             │           Detection
              │             │              │
              ▼             ▼              ▼
       Wrist + Hip      Facial         Contours
       Landmarks       Landmarks          │
              │             │              ▼
              │             ▼       ObjectValidator
              │          solvePnP          │
              │             │              │
              ▼             ▼              ▼
       Wrist Movement   Head Direction   Object Type
              │             │              │
              └─────────────┼──────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │ Baseline.json   │
                    │                 │
                    │ Normal posture  │
                    │ Wrist movement  │
                    │ Static papers   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Student State   │
                    │ Machine         │
                    └────────┬────────┘
                             │
                             ▼
                    Evidence Scoring
                             │
                             ▼
                  Cheating Decision Engine
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
              Individual         Student-to-
               Cheating           Student
               Decision          Interaction
                                     │
                                     ▼
                             Passing State
                               Machine
                                     │
                                     ▼
                            Passing Decision
                                     │
                                     ▼
                               FINAL OUTPUT
                                     │
                                     ▼
                           Annotated Video
                              + Alerts
```

---

# 90. The most important conceptual flow

If you forget everything during your viva, remember this:

```
DETECT
  ↓
TRACK
  ↓
UNDERSTAND BODY
  ↓
UNDERSTAND HEAD
  ↓
FIND OBJECTS
  ↓
REMOVE NORMAL OBJECTS
  ↓
UNDERSTAND TEMPORAL BEHAVIOR
  ↓
ACCUMULATE EVIDENCE
  ↓
MAKE DECISION
  ↓
CHECK INTERACTIONS
  ↓
OUTPUT ALERT
```

---

# 91. A complete example you can tell the evaluator

Suppose the evaluator asks:

> **“Explain what happens when a student cheats.”**
> 

You can answer:

> “First, YOLOv8 detects the student and DeepSORT assigns a persistent ID. For that student, MediaPipe Pose extracts wrist and hip landmarks, while MediaPipe Face Mesh extracts facial landmarks. Using solvePnP, we estimate the student's head orientation and classify the gaze direction. In parallel, the CheatSheetDetector uses HSV thresholding and contour analysis to identify candidate white objects. ObjectValidator evaluates their geometric characteristics to estimate whether they resemble paper or a bottle. The detected information is then passed to the StudentStateMachine, which analyzes the behavior over multiple frames. For example, if the student's hand approaches the pocket, a paper-like object appears, the object is associated with the hand, the student looks down, and the object later disappears under the answer sheet, the state machine moves through several suspicious states and accumulates evidence. Finally, the CheatingDecisionEngine checks whether the required confirmed state and evidence threshold are reached. If two students interact, the InteractionDetector additionally analyzes wrist proximity, sheet possession, ownership transfer, and separation to determine whether a sheet was passed between them.”
> 

That's basically your entire project in one answer.

---

# 92. Your project in 30 seconds

If they suddenly say:

> "Explain your project briefly."
> 

Say:

> **“ExamGuard is an intelligent computer-vision-based surveillance system for detecting suspicious behavior during offline examinations. The system first uses YOLOv8 and DeepSORT to detect and track students with unique IDs. MediaPipe Pose is used to obtain wrist and hip landmarks, while MediaPipe Face Mesh and OpenCV solvePnP are used to estimate head orientation. A baseline learner first learns normal exam behavior and static desk objects. During the actual exam, white-object detection and geometric validation identify suspicious paper-like objects. A temporal state machine combines object presence, hand movement, gaze direction, and pocket movement across multiple frames instead of relying on a single frame. An evidence-based decision engine then determines whether cheating is confirmed. The system also analyzes interactions between students to detect possible cheat-sheet passing.”**
> 

---

# 93. The deepest idea of your project

The project isn't really:

```
"Detect a paper."
```

It isn't:

```
"Detect a person looking down."
```

It isn't even:

```
"Detect a hand near a pocket."
```

The actual idea is:

> **Combine multiple weak visual signals over time to identify a suspicious behavioral pattern.**
> 

That is the central concept of ExamGuard.

```
Weak evidence 1
     +
Weak evidence 2
     +
Weak evidence 3
     +
Temporal sequence
     +
Baseline comparison
     ↓
Stronger decision
```

---

# 94. Your project has 4 levels of evidence

You can remember it this way:

### Level 1 — Person

```
Who?
```

YOLO + DeepSORT

---

### Level 2 — Body

```
Where are their hands?
```

MediaPipe Pose

---

### Level 3 — Head/object

```
Where are they looking?
What are they holding?
```

Face Mesh + solvePnP + CheatSheetDetector

---

### Level 4 — Behavior

```
What happened before and after?
```

State machines + evidence scoring

That's why the architecture is much more than simple object detection.

---

# 95. One final mental model

Imagine your project as an **AI invigilator**:

```
                   AI INVIGILATOR
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
     "WHO?"           "WHERE?"          "WHAT?"
        │                │                │
     YOLO +          MediaPipe +      Object detector
    DeepSORT         Face Mesh
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                    "WHAT NEXT?"
                         │
                         ▼
                  State Machine
                         │
                         ▼
                   "IS IT CHEATING?"
                         │
                         ▼
                  Decision Engine
                         │
                         ▼
                    🚨 ALERT
```

And **BaselineLearner** acts like the invigilator learning the classroom before the exam:

```
"What does normal look like?"
             ↓
       baseline.json
             ↓
"Now I can identify abnormal behavior."
```

That is the complete conceptual picture of your **ExamGuard project**.
