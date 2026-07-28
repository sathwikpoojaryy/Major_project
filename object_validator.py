import cv2
import numpy as np

class ObjectValidator:
    """
    Validates contour characteristics (geometry, aspect ratio, extent)
    to output confidences for 'paper' vs 'bottle' vs 'unknown'.
    """
    def __init__(self):
        pass

    def validate(self, cnt, pw, ph) -> dict:
        """
        cnt: OpenCV contour.
        pw, ph: Bounding box dimensions of the student.
        
        Returns:
            dict: Confidences for 'paper', 'bottle', and 'unknown'.
        """
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0.0
        ar = float(w) / h if h > 0 else 1.0

        paper_conf = 0.0
        bottle_conf = 0.0
        
        # 1. Paper Heuristics (flat, rectangular, moderately square/rectangular aspect ratio)
        # Note: Cheat sheet in hand has extent between 0.33 and 0.80 due to shadows/hand cover
        if extent > 0.32:
            # Paper is usually wide or moderately tall (AR between 0.6 and 3.0)
            if 0.6 <= ar <= 3.2:
                paper_conf = 0.85
                # Boost if extent is high
                if extent > 0.55:
                    paper_conf += 0.10
            else:
                paper_conf = 0.40
        else:
            paper_conf = 0.15

        # 2. Bottle Heuristics (typically taller, lower extent, or specific aspect ratio)
        # AR is very small because bottle is vertically oriented (w < h, ar < 0.65)
        if ar < 0.65:
            bottle_conf = 0.80
            if h / w > 1.5 if w > 0 else True:
                bottle_conf += 0.15
        else:
            # Bottles can sometimes be tilted, but their extent is usually lower if transparent/round
            if extent < 0.30:
                bottle_conf = 0.70
            else:
                bottle_conf = 0.10

        # Normalize confidences
        total = paper_conf + bottle_conf
        if total > 0:
            paper_conf = paper_conf / (total + 0.2)
            bottle_conf = bottle_conf / (total + 0.2)
            unknown_conf = 0.2 / (total + 0.2)
        else:
            paper_conf = 0.1
            bottle_conf = 0.1
            unknown_conf = 0.8

        return {
            "paper": float(round(paper_conf, 2)),
            "bottle": float(round(bottle_conf, 2)),
            "unknown": float(round(unknown_conf, 2))
        }
