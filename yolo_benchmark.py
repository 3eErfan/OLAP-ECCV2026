import cv2
import time
import torch
import torch.nn.functional as F

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.downloads import attempt_download_asset

from tqdm import tqdm
import shutil
from datetime import datetime
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_nms_ultralytics(pred, conf_thres=0.5, iou_thres=0.45, device='cuda'):
    # pred: [1, 84, 8400] or [84, 8400]
    if pred.ndim == 3:
        pred = pred[0]  # [84, 8400]

    # Transpose once: [C, N] → [N, C]
    pred = pred.permute(1, 0)   # [8400, 84]

    # -------------------------------------------------------------
    # Very fast xywh → xyxy (vectorized, no python slicing)
    # -------------------------------------------------------------
    xywh = pred[:, :4]
    cls_scores = pred[:, 4:]

    xyxy = xywh.clone()
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] * 0.5  # x1
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] * 0.5  # y1
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] * 0.5  # x2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] * 0.5  # y2

    # -------------------------------------------------------------
    # Best class per anchor — full GPU reduction
    # -------------------------------------------------------------
    cls_conf, cls = cls_scores.max(dim=1)
    scores = cls_conf

    # -------------------------------------------------------------
    # Confidence filtering (1 vectorized boolean op)
    # -------------------------------------------------------------
    keep = scores > conf_thres
    if keep.sum() == 0:
        return None

    xyxy   = xyxy[keep]
    scores = scores[keep]
    cls    = cls[keep]

    # -------------------------------------------------------------
    # TorchNMS (class-aware): fastest option in Ultralytics
    # -------------------------------------------------------------
    keep_idx = TorchNMS.batched_nms(
        xyxy,
        scores,
        cls,
        iou_threshold=iou_thres,
        use_fast_nms=False if device=='cpu' else True
    )

    # -------------------------------------------------------------
    # Final output: [x1,y1,x2,y2,score,class]
    # -------------------------------------------------------------
    return torch.cat((
        xyxy[keep_idx],
        scores[keep_idx].unsqueeze(1),
        cls[keep_idx].float().unsqueeze(1),
    ), dim=1)

def preprocess_frame(frame, new_shape=640, fp16=False, device='cuda'):
    """
    fast GPU preprocessing pipeline for YOLO:
    - Convert numpy BGR → torch RGB
    - Normalize
    - Letterbox with padding (GPU)
    - CHW, NCHW
    - FP16 optional

    Returns:
        img (torch.Tensor): (1,3,H,W) ready for model
        ratio (float)
        (pad_w, pad_h)  -- for scaling boxes back
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    # If user asked for CUDA but it's not available, fall back to CPU
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    # ------------------------------
    # CPU branch (match preprocess_frame_cpu)
    # ------------------------------
    if device.type == "cpu":
        img, r, (left, top) = preprocess_frame_cpu(frame, new_shape=new_shape, fp16=fp16)

        # keep behavior consistent: CPU path returns float32
        # (fp16 on CPU is usually not helpful and may be unsupported in some ops)
        return img, r, (left, top)



    # ------------------------------
    # GPU branch
    # ------------------------------
    im = torch.from_numpy(frame).to(device)  # GPU tensor

    # BGR → RGB
    im = im[..., [2, 1, 0]]

    # HWC → CHW, uint8 → float32, normalize to 0–1
    im = im.permute(2, 0, 1).contiguous().float() / 255.0
    im = im.unsqueeze(0)   # (1,3,H,W)

    # ------------------------------
    # YOLO Letterbox function on GPU
    # ------------------------------
    _, _, h, w = im.shape

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    new_h, new_w = new_shape

    r = min(new_h / h, new_w / w)
    nh = int(h * r)
    nw = int(w * r)

    # Resize (GPU)
    im_resized = F.interpolate(
        im, size=(nh, nw), mode="bilinear", align_corners=False
    )

    # Padding calculation
    pad_h = new_h - nh
    pad_w = new_w - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad on GPU (use YOLO gray=114/255=0.447 for visual consistency)
    img = F.pad(
        im_resized,
        (left, right, top, bottom),
        value=0.447
    )

    # ------------------------------
    # FP16 (optional)
    # ------------------------------
    if fp16:
        img = img.half()

    return img, r, (left, top)

def preprocess_frame_cpu(frame, new_shape=640, fp16=False):
    """
    CPU-only preprocessing for YOLO:
    - Convert HWC BGR → CHW RGB
    - Normalize
    - Letterbox (resize + pad)
    """

    # Convert numpy to torch CPU
    im = torch.from_numpy(frame).float()  # CPU tensor
    im = im.permute(2, 0, 1)              # HWC -> CHW
    im = im[[2, 1, 0], :, :] / 255.0      # BGR -> RGB + normalize
    im = im.unsqueeze(0)                  # (1, 3, H, W)

    _, _, h, w = im.shape

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    new_h, new_w = new_shape

    r = min(new_h / h, new_w / w)
    nh, nw = int(h * r), int(w * r)

    # Resize
    im_resized = F.interpolate(im, size=(nh, nw),
                               mode="bilinear", align_corners=False)

    # Padding
    pad_h, pad_w = new_h - nh, new_w - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img = F.pad(im_resized, (left, right, top, bottom), value=0.447)

    return img.float(), r, (left, top)

def ensure_model_exists(model_name: str) -> str:
    if not model_name.endswith(".pt"):
        model_name += ".pt"

    local_path = os.path.abspath(model_name)

    # If file already exists and looks valid, use it
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 1000:
        print(f"[OK] Model found: {local_path}")
        return local_path

    print(f"[INFO] Downloading model: {model_name}")
    downloaded = attempt_download_asset(model_name)
    downloaded_abs = os.path.abspath(downloaded)

    # Avoid SameFileError
    if downloaded_abs != local_path:
        shutil.copyfile(downloaded_abs, local_path)

    # Final sanity check
    if not os.path.isfile(local_path) or os.path.getsize(local_path) < 1000:
        raise RuntimeError("Downloaded YOLO model seems corrupted.")

    print(f"[DONE] Model ready at: {local_path}")
    return local_path



class yolo_benchmark:

    def __init__(self, video_path: str, model_path: str, img_size: int, generate_output: bool = False, device: str = 'cpu'):

        self.device = select_device(device)     
        print("Torch:", torch.__version__)
        print("Device:", self.device)

        self.model = AutoBackend(model_path, device=self.device, fp16=False)
        self.model.eval()

        self.IMG_SIZE = img_size
        self.model_name = model_path.split("/")[-1]

        self.cap = cv2.VideoCapture(video_path)

        self.generate_output = generate_output
        if self.generate_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(f"{video_path}_out.mp4", fourcc, 30.0,
                                       (int(self.cap.get(3)), int(self.cap.get(4))))
        else:
            self.out = None

        self.inference_times = []
        self.results_file_name = "benchmark_results.txt"

    def run(self, max_frames=100):

        pbar = tqdm(total=max_frames, desc="running",
                    ncols=80,
                    bar_format='{postfix} {l_bar}{bar}| {n_fmt}/{total_fmt}')

        for _ in range(max_frames):
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or cannot fetch frame.")
                break

            # PREPROCESS
            t0 = time.time()
            img, ratio, (dw, dh) = preprocess_frame(frame, self.IMG_SIZE, fp16=False, device=self.device)
            t1 = time.time()

            # INFERENCE
            with torch.no_grad():
                pred = self.model(img)
            t2 = time.time()

            # NMS
            pred = run_nms_ultralytics(pred[0], device=self.device)
            t3 = time.time()

            # RESCALE
            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
            t4 = time.time()

            # DRAW
            if self.generate_output:
                annotator = Annotator(frame, line_width=2)

                if pred is not None:
                    for *xyxy, conf, cls_id in pred:
                        annotator.box_label(xyxy, "", color=colors(int(cls_id), True))

                self.out.write(annotator.result())
            t5 = time.time()

            # TIMING
            preprocess_ms = (t1 - t0) * 1000
            infer_ms = (t2 - t1) * 1000
            nms_ms = (t3 - t2) * 1000
            annot_ms = (t5 - t4) * 1000

            self.inference_times.append([preprocess_ms, infer_ms, nms_ms, annot_ms])

            pbar.set_postfix({
                "Pre": f"{preprocess_ms:4.1f}",
                "Inf": f"{infer_ms:6.1f}",
                "NMS": f"{nms_ms:4.1f}",
                "Ann": f"{annot_ms:3.1f}",
            })
            pbar.update(1)

        self.cap.release()
        if self.generate_output:
            self.out.release()

        pbar.close()
        # self.show_save_results()

    def show_save_results(self):
        times = np.array(self.inference_times)[10:]
        avg = times.mean(axis=0)

        preprocess_ms, infer_ms, nms_ms, annot_ms = avg
        total_ms = avg.sum()
        fps = 1000 / total_ms

        print(f"\nAverage Times ({self.device}):")
        print(f"Preprocess: {preprocess_ms:.2f} ms")
        print(f"Inference : {infer_ms:.2f} ms")
        print(f"NMS       : {nms_ms:.2f} ms")
        print(f"Annotate  : {annot_ms:.2f} ms")
        print(f"Total     : {total_ms:.2f} ms")
        print(f"FPS       : {fps:.2f}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = self.model_name.replace("_Argoverse.pt", ".pt")
        line = f"{timestamp}, {model_name}, {self.IMG_SIZE}, {preprocess_ms:.2f}, {infer_ms:.2f}, {nms_ms:.2f}, {annot_ms:.2f}, {total_ms:.2f}, {fps:.2f}"

        with open(self.results_file_name, "a") as f:
            f.write(line + "\n")

def run_benchmark(model_name, img_size, results_file_name, video_path="video.mp4", device='cpu', n_frames=100):
    if not os.path.exists(results_file_name):
        with open(results_file_name, "w") as f:
            f.write("Date&Time,Model,InputSize,Preprocess,Inference,NMS,Annotate,Total,FPS\n")
    model_path = ensure_model_exists(model_name)

    benchmark = yolo_benchmark(video_path, model_path, img_size, generate_output=False, device=device)
    benchmark.results_file_name = results_file_name
    benchmark.run(max_frames=n_frames)

if __name__ == "__main__":

    video_path = "video.mp4"
    number_of_frames = 100
    results_file_name = "benchmark_results_M3_mps.txt"

    if not os.path.exists(results_file_name):
        with open(results_file_name, "w") as f:
            f.write("Date&Time,Model,InputSize,Preprocess,Inference,NMS,Annotate,Total,FPS\n")

    for model in ["yolo11n"]:#, "yolo11m", "yolo11x"]:
        model_path = ensure_model_exists(model)
        for img_size in [1920]:#, 416, 640, 1920]:
            print(f"\nRunning benchmark for Model: {model}, Image Size: {img_size}")
            benchmark = yolo_benchmark(video_path, model_path, img_size, generate_output=False, device='cpu')
            benchmark.run(max_frames=number_of_frames)
