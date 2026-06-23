import cv2.legacy as cv2
import pandas as pd
import numpy as np
import time


class Tracker:
    """
    tracker_type:
        0 -> MOSSE
        1 -> KCF
        2 -> CSRT
    """
    def __init__(self, tracker_type: int = 0):
        self.tracker_type = int(tracker_type)

        self.trackers = []
        self.average_fps = 0
        self.frame_count = 0
        self.inference_time = 0

        self.bboxes = None
        self.most_recent_frame = -1

        # Tracking related parameters that are changed from detect function
        self.source_frame_idx = None
        self.pred = None
        self.source_frame = None
        self.need_to_re_initialize = False
        self.is_initialized = False

    def _create_tracker(self):
        """Factory for OpenCV legacy single-object trackers."""
        if self.tracker_type == 0:
            return cv2.TrackerMOSSE_create()
        elif self.tracker_type == 1:
            return cv2.TrackerKCF_create()
        elif self.tracker_type == 2:
            return cv2.TrackerCSRT_create()
        else:
            raise ValueError(
                f"Invalid tracker_type={self.tracker_type}. Use 0 (MOSSE), 1 (KCF), or 2 (CSRT)."
            )

    def initialize(self):
        init_guess = self.pred
        img_a = self.source_frame

        if init_guess is None or len(init_guess) == 0:
            self.bboxes = np.array([]).reshape(0, 4)
            self.trackers = []
            return

        self.bboxes = init_guess[:, :4].copy()                     # xyxy format
        self.bboxes[:, 2] = self.bboxes[:, 2] - self.bboxes[:, 0]  # convert to width
        self.bboxes[:, 3] = self.bboxes[:, 3] - self.bboxes[:, 1]  # convert to height

        self.trackers = []
        for bbox in self.bboxes:
            tracker = self._create_tracker()
            tracker.init(img_a, bbox) #---> x1y1wh format
            self.trackers.append(tracker)
            


    def update(self, new_frame):
        start_time = time.time()

        if self.bboxes is None or len(self.bboxes) == 0:
            return False

        bboxes = self.bboxes.copy()

        # Update bbox data; if tracking fails, keep previous bbox (your original behavior)
        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(new_frame)
            if success:
                bboxes[i] = bbox

        self.bboxes = bboxes

        self.frame_count += 1
        self.inference_time += (time.time() - start_time)
        # avoid divide-by-zero on first frame if something weird happens
        if self.inference_time > 0:
            self.average_fps = self.frame_count / self.inference_time

        return True

    def get_tracked_bboxes(self):
        if self.bboxes is None or len(self.bboxes) == 0:
            return np.array([]).reshape(0, 4)

        # Convert bboxes back to xyxy format
        tracked_bboxes = self.bboxes.copy()
        tracked_bboxes[:, 2] = tracked_bboxes[:, 0] + tracked_bboxes[:, 2]  # x2
        tracked_bboxes[:, 3] = tracked_bboxes[:, 1] + tracked_bboxes[:, 3]  # y2
        return tracked_bboxes