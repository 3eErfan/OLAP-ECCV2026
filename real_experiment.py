import queue
import cv2
import time
import torch
import torch.nn.functional as F

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device, get_cpu_info, get_gpu_info
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import scale_boxes, xyxy2xywh, xywh2xyxy
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.downloads import attempt_download_asset

from tqdm import tqdm
import shutil
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import threading
import copy

from DataLoader import DatasetStreamer
from IoUmodel import IoUPredictionModel, ROMA
from tracker import Tracker as VisualTracker
from yolo_benchmark import preprocess_frame, ensure_model_exists, run_nms_ultralytics

class EXPERIMENT:
    models = []
    model_names = []
    input_sizes = []
    inference_times = []

    model_knob = 1
    input_size_knob = 2

    frameHistory = [] # [[idx, frame], [idx, frame], ... ]
    frame_buffer = None
    frame_idx_buffer = 0
    generate_output_video = False
    save_realtime_detections = True
    streamer_starving = False
    
    VT_LOCK = threading.Lock()

    Currently_Available_Detection = None
    Currently_Available_Detection_Source_Frame_Idx = None

    visual_tracker = VisualTracker()  

    deadlines, bboxes = None, None # for visualization, otherwise only used in IoU model and controller

    control_signal = []
    write_queue = queue.Queue()
    output_video_queue = queue.Queue()
    
    loading_image_required_time = 0.02  # seconds

    stop_watch_detector = None
    n_detections = 0

    def __init__(self,
                generate_output_video: bool = False,
                save_realtime_detections: bool = True, 
                device_priority: list = ['cuda','mps','cpu'], 
                is_offline: bool = False,
                use_ROMA: bool = False,
                is_MOT: bool = False,
                run_extention: str = '',
                run_background_workload: bool = False):
        self.run_extention = run_extention
        self.run_background_workload = run_background_workload
        self.background_workload_size_idx = 0
        self.dataset_streamer = DatasetStreamer(is_MOT=is_MOT)
        self.device = self._pick_device(device_priority)
        self.use_GT_label_instead = False # for debugging purposes, set to True to bypass the detector and use GT labels instead
        self.is_offline = True if self.use_GT_label_instead else is_offline
        
        device = select_device(self.device)
        self.device_name = get_cpu_info() if device.type != 'cuda' else get_gpu_info(device.index)
        self.is_jetson = "Orin" in self.device_name 
        print("Torch:", torch.__version__)
        print("Device:", device)
        self.iou_model = IoUPredictionModel(is_MOT=is_MOT, device=self.device, device_name=self.device_name)

        self.input_sizes  = self.iou_model.input_sizes
        unique_names = self.iou_model.model_names

        # print("Input sizes:", self.input_sizes)
        for model_name in unique_names:
            print("Loading model:", model_name)
            # model_path = ensure_model_exists(model_name)
            model_path = f'./assets/{model_name}_{"MOT" if is_MOT else "Argoverse"}.pt'
            model = AutoBackend(model_path, device=device, fp16=self.is_jetson)
            model.eval()
            self.models.append(model)
            self.model_names.append(model_name)  
        
        self.generate_output_video = generate_output_video
        self.save_realtime_detections = save_realtime_detections
        
        # Set default knobs
        self.constant_model = False
        self.current_inputsize = self.input_sizes[self.input_size_knob]
        self.current_model = self.models[self.model_knob]
        self.background_workload_model = AutoBackend(f'./assets/yolo11n_{"MOT" if is_MOT else "Argoverse"}.pt', device=device, fp16=self.is_jetson)

        self.use_ROMA = use_ROMA
        self.ROMA = ROMA(Detectability=self.iou_model.get_detectability, InferenceTime=self.iou_model.inferenceTime)

    def reset(self):
        """
        Resets the experiment state to initial conditions without reloading
        heavy resources (like PyTorch models).
        """
        # --- Reset Simple State Variables ---
        self.frameHistory = []
        self.frame_buffer = None
        self.frame_idx_buffer = 0
        
        self.Currently_Available_Detection = None
        self.Currently_Available_Detection_Source_Frame_Idx = None
        
        self.deadlines = None
        self.bboxes = None
        self.stop_watch_detector = None

        # --- Reset Queues ---
        self.control_signal = []
        self.write_queue = queue.Queue()
        self.output_video_queue = queue.Queue()

        # --- Reset Sub-Classes --- (except for Argoverse Streamer)
        self.visual_tracker = VisualTracker() 
        if self.use_ROMA:
            self.ROMA.inferenceTime.save_benchmark()
        else:
            self.iou_model.inferenceTime.save_benchmark()
        self.iou_model.reset()
        self.ROMA.reset()

        # --- Reset Knobs / Configuration ---
        # If we are NOT in constant model mode, reset knobs to default class attributes.
        if not self.constant_model:
            self.model_knob = 1  # Reset to default class value
            self.input_size_knob = 2 # Reset to default class value
            
            # Sync the current pointers based on the reset knobs
            if self.input_sizes and len(self.input_sizes) > self.input_size_knob:
                self.current_inputsize = self.input_sizes[self.input_size_knob]
            
            if self.models and len(self.models) > self.model_knob:
                self.current_model = self.models[self.model_knob]

    def _pick_device(self, device_priority: list[str]) -> str:
        for dev in device_priority:
            d = dev.lower().strip()

            # CUDA (allow "cuda" or "cuda:0", etc.)
            if d.startswith("cuda"):
                if torch.cuda.is_available():
                    return str(torch.device(d))  # e.g., "cuda" or "cuda:0"
                continue

            # Apple Silicon MPS
            if d == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return str(torch.device("mps"))
                continue

        # Safe fallback
        return str(torch.device("cpu"))

    def set_constant_model(self, constant_model_idx: int):
        """
        Sets the experiment to use a constant model based on the provided index.
        This method updates the model and input size knobs accordingly.
        """
        self.model_knob = constant_model_idx // len(self.input_sizes)
        self.input_size_knob = constant_model_idx % len(self.input_sizes)
        self.current_inputsize = self.input_sizes[self.input_size_knob]
        self.current_model = self.models[self.model_knob]
        self.constant_model = True
        self.reset()
        # print(f"\n\nConstant model set to: {self.model_names[self.model_knob]} with input size {self.current_inputsize}")

    def controller(self, pred, inference_time_ms):
        inference_time_F = np.ceil(inference_time_ms/33.33).astype(int)  # number of frames at 30 FPS TODO: make sure this read the current fps
        AP_argmax = None
        if pred is not None and len(pred):
            if self.use_ROMA:
                AP_argmax = self.ROMA.get_feedback(
                    pred, 
                    inference_time_ms, 
                    current_model_name=self.model_names[self.model_knob], 
                    current_inputsize=self.current_inputsize)
            else:
                AP_argmax, self.deadlines, self.bboxes = self.iou_model.get_feedback(
                pred, 
                inference_time_ms, 
                current_model_name=self.model_names[self.model_knob], 
                current_inputsize=self.current_inputsize,
                only_update_timetable = self.constant_model)


        if AP_argmax is not None:
            self.model_knob = AP_argmax // len(self.input_sizes)
            self.input_size_knob = AP_argmax % len(self.input_sizes)
        else: # fallback to default
            self.model_knob = 1
            self.input_size_knob = 2

        self.current_inputsize = self.input_sizes[self.input_size_knob]
        self.current_model = self.models[self.model_knob]

    def tracker(self):
        def safe_assignment(whoCalled=None, frame_idx=None): # this is for debugging purposes
            
            if not self.visual_tracker.is_initialized or self.Currently_Available_Detection is None:
                print("\n[0]")
                return
            tmp = self.visual_tracker.get_tracked_bboxes()


            # HARD RULE: only write if shapes match
            if self.Currently_Available_Detection.shape[0] != tmp.shape[0]:
                # skip (or replace the whole thing—see below)
                print(f"\n[2] Skipping bbox assignment at frame {frame_idx} called by {whoCalled} due to shape mismatch: {self.Currently_Available_Detection.shape} vs {tmp.shape}")
                return

            self.Currently_Available_Detection[:, :4] = tmp

        if self.visual_tracker.need_to_re_initialize and not self.visual_tracker.is_initialized:
            with self.VT_LOCK:
                self.visual_tracker.need_to_re_initialize = False
                self.visual_tracker.is_initialized = True
            self.visual_tracker.initialize()
            while self.frameHistory and self.frameHistory[0][0] < self.visual_tracker.source_frame_idx:
                self.frameHistory.pop(0)
            OK = True
            for frame_idx, frame in self.frameHistory:          # start from source frame till the latest frame
                OK = OK and self.visual_tracker.update(frame)
            
            with self.VT_LOCK:
                if self.visual_tracker.need_to_re_initialize:
                    return
                if OK:
                    safe_assignment("INIT", frame_idx)

            return
    
        elif  not self.visual_tracker.need_to_re_initialize and self.visual_tracker.is_initialized: # update for new frame

            frame_idx, frame = self.frameHistory[-1]            # get the latest frame
            if self.visual_tracker.most_recent_frame < frame_idx:
                self.visual_tracker.most_recent_frame = frame_idx
                OK = self.visual_tracker.update(frame)
                if OK:
                    with self.VT_LOCK:
                        if self.visual_tracker.need_to_re_initialize:
                            return
                        safe_assignment("update", frame_idx)
            else:
                time.sleep(0.01)
                return # no new frame to update

        else: # no tracking needed, just load more frames to cache
            return

    def detect(self, frame, frame_idx):

        # For logging and analysis purposes. the "control signal" and its "source frame"
        model_knob, input_size_knob = self.model_knob, self.input_size_knob

        if self.use_GT_label_instead:
            t0, t1, t2, t3, t4 = 0, 0, 0, 0, 0
            pred = self.dataset_streamer.get_frame_GT_label(val=self.cap.val, 
                                                    folder_index=self.cap.folder_index, 
                                                    frame_index=frame_idx)
            # scale GT boxes to absolute pixel values
            if pred is not None and len(pred):
                pred[:, [0,2]] *= frame.shape[1]  # scale x by width
                pred[:, [1,3]] *= frame.shape[0]  # scale y by height 
            self.Currently_Available_Detection = pred.copy() if pred is not None else np.zeros((0,6))
            self.Currently_Available_Detection_Source_Frame_Idx = frame_idx
        else:
            # PREPROCESS
            t0 = time.time()
            img, ratio, (dw, dh) = preprocess_frame(frame, self.current_inputsize, fp16=self.is_jetson, device=self.device)
            t1 = time.time()

            # INFERENCE
            with torch.no_grad():
                pred = self.current_model(img)
            t2 = time.time()

            # NMS
            pred = run_nms_ultralytics(pred[0], device=self.device)
            t3 = time.time()

            # RESCALE
            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
            t4 = time.time()



            with self.VT_LOCK:
                self.Currently_Available_Detection = pred.cpu().numpy().copy() if pred is not None else np.zeros((0,6))
                self.Currently_Available_Detection_Source_Frame_Idx = frame_idx
                # Prepare visual tracker for next round
                self.visual_tracker.need_to_re_initialize = True
                self.visual_tracker.is_initialized = False
                self.visual_tracker.source_frame = frame.copy()
                self.visual_tracker.source_frame_idx = frame_idx
                self.visual_tracker.pred = self.Currently_Available_Detection.copy()


        # TIMING 1
        preprocess_ms = (t1 - t0) * 1000
        infer_ms = (t2 - t1) * 1000
        nms_ms = (t3 - t2) * 1000
        if self.stop_watch_detector is None:
            total_inferenceTime_ms= (t4 - t0) * 1000 if frame_idx > 0 else 100 # TODO: Explain where 100 = 3 * 33.33 comes from
        else:
            total_inferenceTime_ms = (time.time() - self.stop_watch_detector) * 1000
            while self.streamer_starving:
                time.sleep(0.001)
            self.stop_watch_detector = time.time()

        # CONTROLLER
        t5 = time.time()
        if not self.is_offline and not self.constant_model:
            self.controller(pred, total_inferenceTime_ms) 
        t6 = time.time()
        controller_ms = (t6 - t5) * 1000

        self.control_signal.append([
                                    frame_idx,
                                    model_knob,       # not the new model knob
                                    input_size_knob,  # not the new input size knob
                                    preprocess_ms,
                                    infer_ms,
                                    nms_ms,
                                    controller_ms
                                ])
        self.n_detections += 1
    
    def background_workload_detect(self, frame):

        BG_current_inputsize = self.input_sizes[self.background_workload_size_idx]

        # PREPROCESS
        img, ratio, (dw, dh) = preprocess_frame(frame, BG_current_inputsize, fp16=self.is_jetson, device=self.device)


        # INFERENCE
        with torch.no_grad():
            pred = self.background_workload_model(img)

        # NMS
        pred = run_nms_ultralytics(pred[0], device=self.device)
                
    def Stream(self, video_idx=0 ,_video_path=None, fps=30.0, is_val=True):
        is_offline = self.is_offline
        
        successful_stream = False
        while not successful_stream:
            if _video_path is not None:
                self.cap = cv2.VideoCapture(_video_path)
            else:
                self.cap = self.dataset_streamer.cap(val=is_val, folder_index=video_idx, generate_output_video = self.generate_output_video)
                max_frames = self.cap.total_frames

                video_Dir = self.cap.virtual_video_path # path to the source images directory
                if self.constant_model and not self.use_GT_label_instead:
                    detections_dir = video_Dir.replace("/RT_detections",\
                    f"/{'Offline' if is_offline else 'RT'}_detections{'' if is_offline else '_constantModel'}_{self.model_names[self.model_knob]}_{self.current_inputsize}")
                elif self.use_GT_label_instead:
                    detections_dir = video_Dir.replace("/RT_detections", f"/GT_detections")
                elif self.use_ROMA:
                    detections_dir = video_Dir.replace("/RT_detections", f"/RT_detections_ROMA{self.run_extention}")
                else:
                    detections_dir = video_Dir.replace("/RT_detections", f"/RT_detections_{self.device_name}_{self.iou_model.TUNE}{self.run_extention}")
                
                
                if not os.path.exists(detections_dir):
                    os.makedirs(detections_dir)
                else:
                    # If the directory already exists, remove it to start fresh
                    shutil.rmtree(detections_dir)
                    os.makedirs(detections_dir)

                # Retrieve the generated virtual path so the saving logic below doesn't crash
                video_path = os.path.join(detections_dir, self.cap.scene_name + ".mp4")

            frame_width = int(self.cap.get(3))
            frame_height = int(self.cap.get(4))
            max_frames = int(self.cap.get(7))
            print(f"Video properties: {frame_width}x{frame_height} at {fps} FPS, total frames: {max_frames}")

            if self.generate_output_video:
                if "apple" in self.device_name.lower():
                    fourcc = cv2.VideoWriter_fourcc(*"avc1") # <-- "avc1" this format is supported by VSConde on my Macbook
                else:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out = cv2.VideoWriter(video_path, fourcc, fps,
                                        (frame_width, frame_height))
            else:
                self.out = None
                self.output_video_queue.put(None)  # Signal the video drawing thread to exit immediately

            pbar = tqdm(total=max_frames, desc="running",
                        ncols=120,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')

            current_fps = 0
            one_over_fps = 1.0 / fps

            # Infinite loop over frames
            for frame_idx in range(max_frames+1):
                
                tic = time.time()
                target_time = tic + one_over_fps

                ret, frame = self.cap.read()

                if not ret:
                    if frame_idx < max_frames - 1:
                        pbar.close()
                        print(f"   Warning!  Frame {frame_idx} could not be read, stopping early.")
                        successful_stream = False
                        print("Start from the beginning after 3 seconds...")
                        time.sleep(3)
                    else:
                        pbar.close()   
                        print("End of video reached. (Breaking out of frame read loop)")
                        successful_stream = True
                    break
                
                self.frame_idx_buffer = frame_idx
                self.frame_buffer = frame
                self.frameHistory.append([frame_idx, frame])
                
                if is_offline:
                    while self.Currently_Available_Detection_Source_Frame_Idx is None or \
                        self.Currently_Available_Detection_Source_Frame_Idx < frame_idx:
                        time.sleep(0.001)


                if self.generate_output_video:
                    detection_list = []
                    pred = self.Currently_Available_Detection
                    if pred is not None and len(pred):
                        detection_list = pred.tolist()
                    self.output_video_queue.put((frame_idx, detection_list))
                if self.save_realtime_detections:
                    # Check if our cap object has the filename list (it will if it's ArgoverseVideoCapture)
                    if hasattr(self.cap, 'frame_filenames'):
                        fname = self.cap.frame_filenames[frame_idx]
                        current_file_name = os.path.splitext(fname)[0] # Removes ".jpg"
                    else:
                        current_file_name = 'frame'
                    self.save_detection(
                                        current_frame_idx=frame_idx,
                                        detections_dir=detections_dir,
                                        frame_file_name=current_file_name
                                        )
                dot = "●" if self.n_detections % 2 == 0 else " "
                pbar.set_postfix_str(  f"{dot} Knob: {self.model_knob}-{self.input_size_knob} | "+
                                    f"fps: {current_fps:.1f} | "+
                                    f"{self.write_queue.qsize()} f | "+
                                    f"{self.output_video_queue.qsize()} out | "+
                                    f"{self.cap.cached_frames.qsize()} cache")
                pbar.update(1)

                # wait for 1/fps seconds (Exception for first frame cold start)
                #naturally_elapsed_time = (time.time() - tic)
                self.streamer_starving = hasattr(self.cap, 'starvation') and self.cap.starvation
                while  self.streamer_starving or (hasattr(self.cap, 'cached_frames') and (time.time() - tic) < (one_over_fps - self.loading_image_required_time - 0.01)) or is_offline:
                    if self.cap.cached_frames.qsize() >= 50 or self.cap.is_done:
                        self.streamer_starving = hasattr(self.cap, 'starvation') and self.cap.starvation
                        break # enough frames are cached
                    else:
                        t0 = time.time()
                        self.cap.load_to_cache()
                        t1 = time.time()
                        self.loading_image_required_time = 0.9 * self.loading_image_required_time + 0.1 * (t1 - t0)
                    self.streamer_starving = hasattr(self.cap, 'starvation') and self.cap.starvation
                    if self.streamer_starving:
                        pbar.set_postfix_str(  f"Knob: {self.model_knob}-{self.input_size_knob} | "+
                                                f"fps: X.XX | "+
                                                f"{self.write_queue.qsize()} f | "+
                                                f"{self.output_video_queue.qsize()} out | "+
                                                f"{self.cap.cached_frames.qsize()} cache")
                    
                    
                if (time.time() - tic) < one_over_fps:
                    if (time.time() - tic) + 0.0055 < one_over_fps:
                        time.sleep(one_over_fps - (time.time() - tic) - 0.005) # wake up 5ms earlier to busy-wait
                    while time.time() < target_time:
                        # don't sleep, just busy-wait
                        pass
                else:
                    current_fps = 1 / (time.time() - tic)
                    # if not is_offline:
                    #     print(f"   Warning!  Streaming @ {current_fps:.2f} FPS (Target: {fps} FPS)")
                while self.Currently_Available_Detection is None and frame_idx == 2: #(HERE is the Exception for first frame cold start)
                    time.sleep(0.001)
                    pbar.set_postfix_str("Cold Start")

                current_fps = 1 / (time.time() - tic)

        self.cap.release()
        pbar.close()
        self.write_queue.put(None)  # Signal the writer thread to exit
        self.output_video_queue.put(None)  # Signal the video drawing thread to exit
        self.save_control_signal(detections_dir)
        print(f"detection count: {self.n_detections}")
        self.n_detections = 0
        # print tracker fps
        if self.visual_tracker.frame_count > 0:
            print(f"Tracker average FPS: {self.visual_tracker.frame_count / self.visual_tracker.inference_time:.2f}")

    def save_control_signal(self, detections_dir):
        control_signal_file = os.path.join(detections_dir, "control_signal.csv")
        with open(control_signal_file, "w") as f:
            f.write("Current_Frame_Idx, Model_Knob, Input_Size_Knob, preprocess, inference, NMS, controller\n")
            for frame_idx, model_knob, input_size_knob, preprocess, inference, nms, controller in self.control_signal:
                f.write(f"{frame_idx}, {model_knob}, {input_size_knob}, {preprocess:.2f}, {inference:.2f}, {nms:.2f}, {controller:.2f}\n")

    def save_detection(self, current_frame_idx, detections_dir, frame_file_name='frame'):
        pred = self.Currently_Available_Detection
        source_frame_idx = self.Currently_Available_Detection_Source_Frame_Idx        
        file_name = f"{frame_file_name}_{current_frame_idx}_{source_frame_idx}.txt"

        detection_list = []
        if pred is not None and len(pred):
            detection_list = pred.tolist()

        self.write_queue.put((detections_dir, file_name, detection_list))
    
    def draw_save_video(self):
        '''
        return of this function determines whether to continue or stop the video drawing thread
        1. If it returns False, the thread continues running.
        2. If it returns True, the thread stops.
        '''
        STOP = False

        # DRAW
        time.sleep(0.01)

        t5 = time.time()

        if self.output_video_queue.empty():
            return STOP # No frames to process, continue

        # from queue
        task = self.output_video_queue.get()

        if task is None:
            STOP = True  # Exit signal received
            return STOP

        frame_idx, pred = task
        frame = self.cap.streamer.get_frame(self.cap.val, self.cap.folder_index, frame_idx)

        annotator = Annotator(frame, line_width=2)
        
        # if self.bboxes is not None:
        #     bboxes, matched_mask, unmatched_old_mask, unmatched_new_mask = self.bboxes
        #     deadlines = self.deadlines

        
        #     for xyxy, d in zip(xywh2xyxy(bboxes[matched_mask]), deadlines):
        #         annotator.box_label(xyxy, f"{d}", color=colors(int(7), True))
        #     for xyxy in xywh2xyxy(bboxes[unmatched_old_mask]):
        #         annotator.box_label(xyxy, f"FP", color=colors(int(7), True))
        #     for xyxy in xywh2xyxy(bboxes[unmatched_new_mask]):
        #         annotator.box_label(xyxy, f"FN", color=colors(int(6), True))

        if pred is not None and len(pred):
            for *xyxy, conf, cls in pred:
                annotator.box_label(xyxy, f"{int(cls)}", color=colors(int(0), True))

            for model_knob in range(len(self.model_names)):
                for input_size_knob in range(len(self.input_sizes)):
                    color=colors(int(2), True)
                    if model_knob == self.model_knob and input_size_knob == self.input_size_knob:
                        color=colors(int(7), True)
                    x = 170 * model_knob
                    y = 25 * input_size_knob
                    current_model_and_size = f"{self.model_names[model_knob]}_{self.input_sizes[input_size_knob]}"
                    annotator.box_label([25+x,50+y,30+x,55+y], f"{current_model_and_size}", color=color)


        self.out.write(annotator.result())
        t6 = time.time()

        # TIMING 2
        annot_ms = (t6 - t5) * 1000

        return STOP




def main_stream_detect(video_index=0, is_val=True):
    """Simple main that runs Stream and detect in parallel threads.

    This is intentionally minimal: it creates a `yolo_benchmark` instance,
    starts the `Stream` method in one thread and runs a lightweight detector
    loop in another thread that calls `detect()` for each new frame.
    """

    def stream_thread():
        if hasattr(os, 'nice'): 
            # Add 10 to current niceness (max is usually 19, default is 0)
            # Higher number = Lower Priority
            try:
                os.nice(2) 
            except OSError:
                print("Failed to change process niceness.")
                pass # Might fail if permissions are strict

        EXP.Stream(video_idx=video_index, is_val=is_val)

    def detect_thread():
        
        if hasattr(os, 'nice'): 
            # Add 10 to current niceness (max is usually 19, default is 0)
            # Higher number = Lower Priority
            try:
                os.nice(0) 
            except OSError:
                print("Failed to change process niceness.")
                pass # Might fail if permissions are strict

        last_idx = -1
        while True:
            idx = getattr(EXP, "frame_idx_buffer", -1)
            if idx != last_idx and getattr(EXP, "frame_buffer", None) is not None:
                try:
                    frame = EXP.frame_buffer.copy()
                except Exception:
                    frame = copy.deepcopy(EXP.frame_buffer)
                EXP.detect(frame, idx)
                last_idx = idx

            # stop when streamer finished and no new frames
            if not t_stream.is_alive() and idx == last_idx:
                break
            time.sleep(0.01)

    def tracker_thread():
                # 1. Lower priority (Increase niceness)
        if hasattr(os, 'nice'): 
            # Add 10 to current niceness (max is usually 19, default is 0)
            # Higher number = Lower Priority
            try:
                os.nice(1) 
            except OSError:
                print("Failed to change process niceness.")
                pass # Might fail if permissions are strict

        while True:
            EXP.tracker()
            # stop when streamer finished and no new frames
            if not t_stream.is_alive():
                break

    def file_writer_worker():
        """
        Runs in the background, constantly pulling tasks from the queue 
        and writing them to disk.
        """
        # 1. Lower priority (Increase niceness)
        if hasattr(os, 'nice'): 
            # Add 10 to current niceness (max is usually 19, default is 0)
            # Higher number = Lower Priority
            try:
                os.nice(10) 
            except OSError:
                print("Failed to change process niceness.")
                pass # Might fail if permissions are strict
        
        def file_writer():
            STOP = False

            if EXP.write_queue.empty():
                return STOP  # No tasks to process, continue

            # Get data from queue (blocks until data is available)
            task = EXP.write_queue.get()

            if task is None:
                STOP = True
                return STOP  # Exit signal received
            
            # Unpack the task
            detections_dir, file_name, detection_list = task

            try:
                # Create directory if it doesn't exist
                # (Doing this here offloads the OS check from the main thread)
                os.makedirs(detections_dir, exist_ok=True)

                detection_file = os.path.join(detections_dir, file_name)
                
                # Prepare the string content
                txtout = ""
                if detection_list:
                    for *xyxy, conf, cls in detection_list:
                        x1, y1, x2, y2 = xyxy
                        txtout += f"{int(cls)} {int(x1)} {int(y1)} {int(x2)} {int(y2)} {conf:.2f}\n"

                # Write to file
                with open(detection_file, "w") as f:
                    f.write(txtout)
                    
            except Exception as e:
                print(f"Error writing detection file: {e}")
            finally:
                # Mark task as done
                EXP.write_queue.task_done()

            return STOP


        STOP_draw = not EXP.generate_output_video
        STOP_write = False
        while True:
            time.sleep(0.01)

            if not STOP_draw and EXP.generate_output_video:
                STOP_draw  = EXP.draw_save_video()

            if not STOP_write:
                STOP_write = file_writer()

            if STOP_draw and STOP_write:
                break

        if EXP.generate_output_video:
            EXP.out.release()

    def background_workload_thread():
        
        if hasattr(os, 'nice'): 
            # Add 10 to current niceness (max is usually 19, default is 0)
            # Higher number = Lower Priority
            try:
                os.nice(1) 
            except OSError:
                print("Failed to change process niceness.")
                pass # Might fail if permissions are strict

        last_idx = -1
        while True:
            idx = getattr(EXP, "frame_idx_buffer", -1)
            if idx != last_idx and getattr(EXP, "frame_buffer", None) is not None:
                try:
                    frame = EXP.frame_buffer.copy()
                except Exception:
                    frame = copy.deepcopy(EXP.frame_buffer)
                EXP.background_workload_detect(frame)
                last_idx = idx

            # stop when streamer finished and no new frames
            if not t_stream.is_alive() and idx == last_idx:
                break
            time.sleep(0.01)


    t_stream = threading.Thread(target=stream_thread, name="stream", daemon=False, args=())
    t_detect = threading.Thread(target=detect_thread, name="detect", daemon=True)
    writer_thread = threading.Thread(target=file_writer_worker, name="writer", daemon=True)
    
    t_stream.start()
    t_detect.start()
    writer_thread.start()

    if EXP.run_background_workload:
        t_background = threading.Thread(target=background_workload_thread, name="background", daemon=True)
        t_background.start()

    if not EXP.is_offline:
        t_tracker = threading.Thread(target=tracker_thread, name="tracker", daemon=True)
        t_tracker.start()

    t_stream.join()
    t_detect.join()
    writer_thread.join()
    if not EXP.is_offline:
        t_tracker.join()
    if EXP.run_background_workload:
        t_background.join()
        
    EXP.reset()


if __name__ == "__main__":

    is_VAL = False
                    

    EXP = EXPERIMENT(
                generate_output_video = False, 
                is_MOT = False,
                # run_extention = "ARISE"
                )

    video_list = EXP.dataset_streamer.data_map['val' if is_VAL else 'train']


        
    for idx in range(len(video_list)):

        # if  'MOT17-13-DPM' not in str(video_list[idx]):
        #     continue

        for ROMA_flag in [False, True]:
            print(f"\n\n===== Video {idx} | ROMA: {ROMA_flag} ")
            EXP.use_ROMA = ROMA_flag
            main_stream_detect(video_index=idx, is_val=is_VAL)
            time.sleep(1) # cool down period
        
    # for idx in range(len(video_list)):

    #     # if  'MOT17-13-DPM' not in str(video_list[idx]):
    #     #     continue
    
    #     # for model_idx in range(len(EXP.iou_model.input_sizes) * len(EXP.iou_model.model_names) ) : 
    #     #     if model_idx == len(EXP.iou_model.input_sizes)+2:
    #     #         continue
    #     # for model_idx in [ len(EXP.iou_model.input_sizes)+2 ]: # <-- this is index of YOLO11m_640 regardless of device
    #     for model_idx in [ len(EXP.iou_model.input_sizes)+5 ]: # <-- this is index of YOLO11m_1920 regardless of device
            
    #         EXP.set_constant_model(model_idx)

    #         print(f"\n\n===== Video {idx} | Constant Model: {EXP.model_names[EXP.model_knob]} | Input Size: {EXP.current_inputsize}")
    #         main_stream_detect(video_index=idx, is_val=is_VAL)
    #         time.sleep(1) # cool down period
