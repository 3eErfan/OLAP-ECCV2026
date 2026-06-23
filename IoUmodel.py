import numpy as np
import joblib
from scipy.optimize import linear_sum_assignment
from ultralytics.utils.ops import xyxy2xywh, xywh2xyxy

from yolo_benchmark import run_benchmark
from pathlib import Path
from typing import Dict, Union, Optional
import pandas as pd
import os

class IoUPredictionModel:
    
    def __init__(self, is_MOT = True, device='cpu', device_name=None, TUNE = 0.5, use_ARISE = False):
        self.device = device
        is_jetson = "Orin" in device_name 
        self.TUNE = TUNE # tuning parameter to balance speed and accuracy Higher TUNE -> more emphasis on speed
        self.use_ARISE = use_ARISE

        # scaler = joblib.load(f'./assets/scaler_hist_{"MOT" if is_MOT else "ArgoVerse"}_train.pkl')
        # model = joblib.load(f'./assets/GLM_v2_model_hist_{"MOT" if is_MOT else "ArgoVerse"}_train.pkl')
        scaler = joblib.load(f'./assets/scaler_hist_{"MOT" if is_MOT else "ArgoVerse"}_RTGT.pkl')
        model = joblib.load(f'./assets/GLM_v2_model_hist_{"MOT" if is_MOT else "ArgoVerse"}_RTGT.pkl')
        

        _scaler = np.array(list(scaler.values()))
        mu = np.concatenate([_scaler[:, 0], np.zeros(2)])
        std = np.concatenate([_scaler[:, 1], np.ones(2)])

        W_new = model["W"] / std[None, :]          
        b_new = model["b"] - np.dot(W_new, mu)        

        self.W = W_new
        self.b = b_new

        self.softplus = lambda x: np.log1p(np.exp(x))
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

        self.previous_detection = None  
        self.previous_inferenceTime = 3
        self.previous_X = None

        self.input_sizes  = [288, 416, 640, 1120, 1440] if is_jetson else [288, 416, 640, 1120, 1440, 1920]
        self.model_names = ["yolo11n", "yolo11m"] if is_jetson else ["yolo11n", "yolo11m"]#, "yolo11x"]
        required_keys = []
        for i, name in enumerate(self.model_names):
            for j, size in enumerate(self.input_sizes):
                key = f"{name}_{size}"
                required_keys.append(key)

        self.get_detectability = Detectability(f'./assets/{"MOT" if is_MOT else "Argoverse"}_detectability_params.txt', required_keys=required_keys).get_detectability
        self.inferenceTime = InferenceTime(device=self.device, required_keys=required_keys)

        benchmark_run_list = []
        self.model_dic = {}
        for i, name in enumerate(self.model_names):
            for j, size in enumerate(self.input_sizes):
                idx = i * len(self.input_sizes) + j
                self.model_dic[idx] = [name, size]

                key = f"{name}_{size}"
                # check if key exists in TIME_TABLE, else add to run required benchmark
                if key not in self.inferenceTime.TIME_TABLE:
                    print(f"Warning: {key} not found in TIME_TABLE. Please run benchmark to add it.")
                    benchmark_run_list.append([i, j])  # store indices to run benchmark later
        for i, j in benchmark_run_list:
            name = self.model_names[i]
            size = self.input_sizes[j]
            key = f"{name}_{size}"
            print(f"Running benchmark for missing model: {key}")
            run_benchmark(
                model_name= "./assets/" + name + "_" + ("MOT" if is_MOT else "Argoverse"),
                img_size= size,
                results_file_name=f"benchmark_{self.device}.csv",
                device=self.device,
            )
        self.inferenceTime = InferenceTime(device=self.device, required_keys=required_keys)

    def reset(self):
        self.previous_detection = None  
        self.previous_inferenceTime = 3
        self.previous_X = None

    def softplus_np(self, x):
        # stable softplus: log(1 + exp(-|x|)) + max(x, 0)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def predict(self, X, age):

        # Linear transformation
        Z = np.dot(X, self.W.T) + self.b

        # Apply softplus to first output and sigmoid to second output
        beta  = self.sigmoid(Z[:, 0])             # amplitude in (0,1), from a2_raw
        alpha = self.softplus_np(Z[:, 1]) + 1e-6     # decay > 0, from a3_raw

        log_term = -age * alpha / 100.0
        log_term = np.clip(log_term, -50, 50)
        Iou = beta * np.exp(log_term)

        return Iou
    
    def ARISE_base_deadline_predictor(self, feature_iou, feature_age, IoU_threshold=0.5):
        """
        ARISE baseline predictor based on feature IoU degradation.
        """
        pred_d = (1 - IoU_threshold)  *   feature_age / (1 - feature_iou + 1e-6) 
    
        return np.clip(pred_d, 0, 30).astype(int)

    def predict_deadline(self, X, iou):
        """
        Given X and IoU, compute the implied age from:
            IoU = beta * exp(-alpha * age / 100)
        """
        # Xsub = rows of X with X[:, 6] (aspect ratio) less than 1.0
        # Xsub = X[X[:, 6] < 1.0]
        # Xother = X[X[:, 6] >= 1.0]

        # # sort Xsub by X[0] (area) in descending order (larger objects first)
        # Xsub = Xsub[np.argsort(-Xsub[:, 0])]
        
        # # concatenate back with Xother (which are not sorted)
        # X = np.concatenate([Xsub, Xother], axis=0)

        Z = np.dot(X, self.W.T) + self.b

        # matches forward() transformation
        beta  = self.sigmoid(Z[:, 0])             # amplitude in (0,1), from a2_raw
        alpha = self.softplus_np(Z[:, 1]) + 1e-6     # decay > 0, from a3_raw

        ratio = np.clip(iou / beta, 1e-9, 1.0)
        
        age = -(100.0 / alpha) * np.log(ratio)
        return np.ceil(age).astype(int) # return integer number of frames
    
    def runtime_feedback(self, bboxes_old, bboxes_new, new_inferenceTime, IoU_threshold=0.75):
        """
        Returns:
            deadline_pred: (N_matched,) array of predicted deadlines (in frames) for matched boxes
            ... 
        """
        reordered_new, matched_mask, matched_old_mask_per_old,\
        unmatched_old_mask, unmatched_new_mask, giou_all = Hungarian(bboxes_old, bboxes_new)

        bboxes_B = reordered_new[matched_mask]
        bboxes_A = bboxes_old[matched_old_mask_per_old]
        giou = giou_all[matched_mask]

        new_X_features, new_iou, new_giou = pre_process_numpy(bboxes_B, bboxes_A, self.previous_inferenceTime, giou)
    
        # model training X, y pairs: [X = (previous_X, new_inferenceTime), Y = (new_iou, new_giou)]
        # The problem is that the order of X and Y rows must match (relating to matching objects) TODO: complete this part
        # define a queue to store the X, y 

        self.previous_inferenceTime = new_inferenceTime
        self.previous_X = new_X_features
        if self.use_ARISE:
            deadline_pred = self.ARISE_base_deadline_predictor(new_iou, new_X_features[:, -1], IoU_threshold - 0.25)
        else:
            deadline_pred = self.predict_deadline(new_X_features, IoU_threshold) # len = number of matched boxes

        # return deadline_pred, sum(unmatched_new_mask), sum(unmatched_old_mask)
        return deadline_pred, (reordered_new, matched_mask, unmatched_old_mask, unmatched_new_mask)

    def AP_forecast(self, inference_time_curr, AVG_Inferencetime, Detectability, InferenceTimes_F, deadlines):
        '''
        Detectability_N: (12,) array of number of objects detectable by each model
        InferenceTime_F: (12,) array of inference time in frames for each model
        deadlines: (M,) array of deadlines for each detected object

        returns:
            AP_scores: (12,) array of AP scores for each model
        '''
        n_objects = len(deadlines)
        n_models = len(InferenceTimes_F)
        
        if n_objects == 0:
            return Detectability / InferenceTimes_F  # avoid division by zero


        AP_scores_1 = np.zeros_like(InferenceTimes_F, dtype=float)

        # Average AP for frames that still uses current model's detection output
        for model_idx in range(n_models):
            for age in range(inference_time_curr, inference_time_curr + InferenceTimes_F[model_idx]):
                AP_scores_1[model_idx] += np.sum(deadlines >= age) / n_objects
                # AP_scores_1[model_idx] += 9 * (deadlines[0] >= age)
                # AP_scores_1[model_idx] /= n_objects #+ 9
            AP_scores_1[model_idx] /= InferenceTimes_F[model_idx]
        
        # Average AP for frames that use new model's detection output
        AP_scores_2 = np.zeros_like(InferenceTimes_F, dtype=float)
        for model_idx in range(n_models):
            for age in range(InferenceTimes_F[model_idx], InferenceTimes_F[model_idx]+AVG_Inferencetime):
                AP_scores_2[model_idx] += np.sum(deadlines >= age) / n_objects
                # AP_scores_2[model_idx] += 9 * (deadlines[0] >= age)
                # AP_scores_2[model_idx] /= n_objects #+ 9
        AP_scores_2 /= AVG_Inferencetime
        AP_scores_2 *= Detectability

        # AP_scores = (AP_scores_1 * self.TUNE + AP_scores_2 * (1 - self.TUNE)) / (InferenceTimes_F + AVG_Inferencetime)
        AP_scores_1 /= np.mean(AP_scores_1) if np.mean(AP_scores_1) != 0 else 1  # normalize
        AP_scores_2 /= np.mean(AP_scores_2) if np.mean(AP_scores_2) != 0 else 1  # normalize
        
        AP_scores = AP_scores_1*self.TUNE  +  AP_scores_2*(1 - self.TUNE)
        
        # DEBUG: print AP scores in two columns
        # print("Model\tAP_scores_1\tAP_scores_2\tFinal_AP_scores")
        # for model_idx in range(n_models):
        #     model_name, input_size = self.model_dic[model_idx]
        #     print(f"{model_name}_{input_size}\t{AP_scores_1[model_idx]:.4f}\t{AP_scores_2[model_idx]:.4f}\t{AP_scores[model_idx]:.4f}")

        return AP_scores
        
    def get_feedback(self, pred, inference_time_ms, current_model_name, current_inputsize, only_update_timetable = False):
        
        self.inferenceTime.update_inference_time(current_model_name, current_inputsize, inference_time_ms)
        if only_update_timetable:
            return None, None, None

        inference_time_F = np.ceil(inference_time_ms * self.inferenceTime.fps / 1000.0).astype(int)  # in frames
        bboxes_new = xyxy2xywh(pred[:, :4]).cpu().numpy()
        areas_new = bboxes_new[:,2] * bboxes_new[:,3]  # including not matched boxes
        model = f'{current_model_name}_{current_inputsize}'
        Detectability = self.get_detectability(model, areas_new) # total number of objects detectable by each model (12,)
        InferenceTime_F = self.inferenceTime.get_inference_time()  # inference time in frames for each model (12,)

        deadlines, bboxes = None, None
        if self.previous_detection is not None and pred is not None:
            deadlines, bboxes = self.runtime_feedback( #                  <-- Feedback
                bboxes_new,
                self.previous_detection,
                new_inferenceTime= inference_time_F
            )

        if deadlines is not None and len(deadlines) > 0:
            AP_scores = self.AP_forecast(
                inference_time_curr= inference_time_F,
                AVG_Inferencetime= self.inferenceTime.AVERAGE_inference_time,
                Detectability= Detectability,
                InferenceTimes_F= InferenceTime_F,
                deadlines= deadlines
            )
        else:
            AP_scores = None
        
        self.previous_detection = bboxes_new.copy() # store for next cycle

        return np.argmax(AP_scores) if AP_scores is not None else None, deadlines, bboxes
    
def GIoU(bboxes_new, bboxes_old):
    # Unpack
    x, y, w, h = bboxes_new[:,0], bboxes_new[:,1], bboxes_new[:,2], bboxes_new[:,3]
    x_old, y_old, w_old, h_old = bboxes_old[:,0], bboxes_old[:,1], bboxes_old[:,2], bboxes_old[:,3]

    # Vectorized intersection
    x_left   = np.maximum(x, x_old)
    y_top    = np.maximum(y, y_old)
    x_right  = np.minimum(x + w, x_old + w_old)
    y_bottom = np.minimum(y + h, y_old + h_old)

    inter_w = np.maximum(0.0, x_right - x_left)
    inter_h = np.maximum(0.0, y_bottom - y_top)
    intersection_area = inter_w * inter_h

    # Areas
    box1_area = w * h
    box2_area = w_old * h_old

    # Union (avoid division by zero)
    union_area = box1_area + box2_area - intersection_area
    union_area = np.maximum(union_area, 1e-12)

    # Smallest enclosing box
    enclosing_x_left   = np.minimum(x, x_old)
    enclosing_y_top    = np.minimum(y, y_old)
    enclosing_x_right  = np.maximum(x + w, x_old + w_old)
    enclosing_y_bottom = np.maximum(y + h, y_old + h_old)
    enc_w = np.maximum(0.0, enclosing_x_right - enclosing_x_left)
    enc_h = np.maximum(0.0, enclosing_y_bottom - enclosing_y_top)
    enclosing_area = enc_w * enc_h
    enclosing_area = np.maximum(enclosing_area, 1e-12)

    # IoU and GIoU
    iou = intersection_area / union_area
    giou = iou - ((enclosing_area - union_area) / enclosing_area)
    return giou, iou

def pre_process_numpy(bboxes_new, bboxes_old, feature_age, GIoU_list, W=1920, H=1200):
    """
    bboxes_new: (N,4) array [x,y,w,h] for current frame
    bboxes_old: (N,4) array for previous frame (for motion)
    feature_age: (N,) array inference time of the new detection (in frames)

    Returns an (N, 11) matrix with features ordered as 
    ['area',
    '|V|',
    'V_dir',
    'dX',
    'dx/w',
    'dy/h',
    'aspect_ratio',
    'R',
    'theta',
    'feature_giou',
    'feature_age']
    """

    # Unpack
    x, y, w, h = bboxes_new[:,0], bboxes_new[:,1], bboxes_new[:,2], bboxes_new[:,3]
    x_old, y_old, w_old, h_old = bboxes_old[:,0], bboxes_old[:,1], bboxes_old[:,2], bboxes_old[:,3]

    # 1. sqrt(Area)
    area = np.sqrt(w * h)

    deltaX_0 = x - x_old
    deltaX_1 = y - y_old

    # 2. |V|
    Vmag = np.sqrt(deltaX_0**2 + deltaX_1**2) / feature_age

    # 3. direction
    Vdir = np.arctan2(deltaX_1, deltaX_0)

    # 4. dX (normalized total displacement)
    dX = (np.abs(deltaX_0)/w + np.abs(deltaX_1)/h) / feature_age

    # 5. dx/w
    dxw = (np.abs(deltaX_0)/w) / feature_age

    # 6. dy/h
    dyh = (np.abs(deltaX_1)/h) / feature_age

    # 7. Aspect ratio
    aspect_ratio = w / h

    cx = x + w/2
    cy = y + h/2

    # 8. distance to center normalized
    R = np.sqrt((cx - W/2)**2 + (cy - H/2)**2) / np.sqrt(W**2 + H**2)

    # 9. angle of center point
    theta = np.arctan2(cy, cx)

    # 10. feature_giou
    giou, iou = GIoU(bboxes_new, bboxes_old)
    # giou = GIoU_list

    # 11. feature_age (given)
    age = feature_age * np.ones_like(giou)

    # Stack in order
    features = np.stack([
        area, Vmag, Vdir, dX, dxw, dyh,
        aspect_ratio, R, theta,
        giou, age
    ], axis=1)

    return features, iou, giou

def Hungarian(bboxes_old, bboxes_new):
    """
    bboxes_old: (M, 4)
    bboxes_new: (N, 4)

    Returns:
        reordered_new: (M + U, 4) array, where:
          - rows 0..M-1 correspond to old bboxes:
                matched new bbox if matched,
                [-1, -1, -1, -1] if not matched.
          - rows M.. end are unmatched new bboxes.

        matched_mask:      (M + U,) bool
            True where row corresponds to a matched old<->new pair
            (only possible in rows 0..M-1).

        unmatched_old_mask:(M + U,) bool
            True for rows 0..M-1 where that old bbox is unmatched.

        unmatched_new_mask:(M + U,) bool
            True for rows M..end (unmatched new bboxes).

        giou_all:          (M + U,) float32
            GIoU scores for matched rows (0..M-1),
            -1.0 for unmatched old rows (0..M-1 with no match),
            -1.0 for all unmatched new rows (M..end).
    """

    def GIoU_cost_matrix(bboxes_new, bboxes_old):
        """
        bboxes_new: (N,4) array of new bboxes
        bboxes_old: (M,4) array of old bboxes

        Returns an (N,M) cost matrix based on GIoU (minimizing -GIoU).
        """
        N = bboxes_new.shape[0]
        M = bboxes_old.shape[0]

        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for i in range(N):
            bbox_new_i = np.repeat(bboxes_new[i:i+1, :], M, axis=0)
            giou,_ = GIoU(bbox_new_i, bboxes_old)   # shape (M,)
            cost_matrix[i, :] = -giou             # minimize -GIoU -> maximize GIoU

        return cost_matrix

    M = bboxes_old.shape[0]
    N = bboxes_new.shape[0]

    # Edge cases: no old or no new bboxes
    if M == 0 and N == 0:
        return (
            np.empty((0, 4), dtype=float),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=bool),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    if M == 0:
        # No old boxes → all new are unmatched new
        reordered_new = bboxes_new.copy()
        total_rows = N
        matched_mask = np.zeros(total_rows, dtype=bool)
        unmatched_old_mask = np.zeros(total_rows, dtype=bool)
        unmatched_new_mask = np.ones(total_rows, dtype=bool)
        giou_all = np.full(total_rows, -1.0, dtype=np.float32)
        unmatched_old_indices = np.empty((0,), dtype=int)
        unmatched_new_indices = np.arange(N, dtype=int)
        return (reordered_new, matched_mask, unmatched_old_mask,
                unmatched_new_mask, giou_all,
                unmatched_old_indices, unmatched_new_indices)

    if N == 0:
        # No new boxes → all old rows are unmatched, filled with -1
        aligned = np.full((M, 4), -1.0, dtype=np.float32)
        reordered_new = aligned
        total_rows = M
        matched_mask = np.zeros(total_rows, dtype=bool)
        unmatched_old_mask = np.ones(total_rows, dtype=bool)
        unmatched_new_mask = np.zeros(total_rows, dtype=bool)
        giou_all = np.full(total_rows, -1.0, dtype=np.float32)
        unmatched_old_indices = np.arange(M, dtype=int)
        unmatched_new_indices = np.empty((0,), dtype=int)
        return (reordered_new, matched_mask, unmatched_old_mask,
                unmatched_new_mask, giou_all,
                unmatched_old_indices, unmatched_new_indices)

    # Compute cost matrix using GIoU (N x M)
    cost_matrix = GIoU_cost_matrix(bboxes_new, bboxes_old)

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind: indices in bboxes_new
    # col_ind: indices in bboxes_old

    # GIoU scores of matched pairs (undo the minus sign)
    giou_matched = -cost_matrix[row_ind, col_ind]  # length = min(N,M)

    # Which old boxes are matched
    all_old_indices = np.arange(M)
    matched_old_mask_per_old = np.zeros(M, dtype=bool)
    matched_old_mask_per_old[col_ind] = True
    unmatched_old_indices = all_old_indices[~matched_old_mask_per_old]

    # Fill aligned rows (per old bbox index)
    aligned = np.full((M, 4), -1.0, dtype=bboxes_new.dtype)
    giou_aligned = np.full((M,), -1.0, dtype=np.float32) # remains -1 for objects disappeared or not detected

    # For each match: new_idx -> old_idx
    for new_idx, old_idx, giou_score in zip(row_ind, col_ind, giou_matched):
        aligned[old_idx] = bboxes_new[new_idx]
        giou_aligned[old_idx] = giou_score

    # Find unmatched new bboxes
    all_new_indices = np.arange(N)
    matched_new_mask_per_new = np.zeros(N, dtype=bool)
    matched_new_mask_per_new[row_ind] = True
    unmatched_new_indices = all_new_indices[~matched_new_mask_per_new]
    unmatched_new = bboxes_new[unmatched_new_indices]

    # Concatenate: first aligned (per old), then unmatched new
    if unmatched_new.size > 0:
        reordered_new = np.concatenate([aligned, unmatched_new], axis=0)
        giou_unmatched_new = np.full(unmatched_new.shape[0], -1.0, dtype=np.float32) # -1 for newly appeared objects
        giou_all = np.concatenate([giou_aligned, giou_unmatched_new], axis=0)
    else:
        reordered_new = aligned
        giou_all = giou_aligned

    # --- Build masks over reordered_new rows ---
    total_rows = reordered_new.shape[0]

    matched_mask = np.zeros(total_rows, dtype=bool)
    unmatched_old_mask = np.zeros(total_rows, dtype=bool)
    unmatched_new_mask = np.zeros(total_rows, dtype=bool)

    # First M rows correspond to old boxes
    matched_mask[:M] = matched_old_mask_per_old
    unmatched_old_mask[:M] = ~matched_old_mask_per_old

    # Rows after M are unmatched new boxes, if any
    if total_rows > M:
        unmatched_new_mask[M:] = True

    return (reordered_new,
            matched_mask,
            matched_old_mask_per_old,
            unmatched_old_mask,
            unmatched_new_mask,
            giou_all)



# this comes from Offline Detectability analysis
class Detectability:

    def __init__(self, detectability_params_path, required_keys=None):
        self.required_keys = required_keys
        self.CUTS, self.P = self.load_params(detectability_params_path)
        self.total_area = 1920 * 1080 if "MOT" in detectability_params_path else 1920 * 1200
        

    def load_params(self, path):
        namespace = {"np": np}
        with open(path, "r") as f:
            code = f.read()

        exec(code, namespace)

        cut1 = namespace["cut1"]
        cut2 = namespace["cut2"]
        P = namespace["P"]

        if self.required_keys is not None:
            P = {k: v for k, v in P.items() if k in self.required_keys}

        return (cut1, cut2), P


    def _get_probability_ratio(self, M_now):
        DEN = 1 / (self.P[M_now] + 1e-8) # shape (3,)
        NUM = np.array(list(self.P.values()))  # shape (12, 3)

        return NUM * DEN[None, :]  # shape (12, 3)
    
    def _get_size_counts(self, areas):
        """
        areas: (N,) array of area of detected boxes
        """
        percentages = areas / self.total_area * 100.0
        small_count = np.sum(percentages < self.CUTS[0])
        medium_count = np.sum((percentages >= self.CUTS[0]) & (percentages < self.CUTS[1]))
        large_count = np.sum(percentages >= self.CUTS[1])

        return np.array([small_count, medium_count, large_count]).astype(int) # array of shape (3,)

    def get_detectability(self, M_now, areas):
        '''
        M_now: current detection model
        sqrt_areas: (N,) array of sqrt(area) of detected boxes

        returns:
            detectability: (12,) array of relative detectability scores for each model
        '''
        size_counts = self._get_size_counts(areas)  # shape (3,)
        
        prob_ratio = self._get_probability_ratio(M_now)  # shape (12, 3)

        detectability = np.sum(prob_ratio * size_counts[None, :], axis=1)  # shape (12,)

        # normalize by detectability of current model
        N_M_now = len(areas) #np.sum(self.P[M_now] * size_counts)
        detectability /=  (N_M_now if N_M_now > 0 else 1.0)

        return detectability

class InferenceTime:    
    def __init__(self,fps=30, device = 'cpu', required_keys=None):
        self.fps = fps
        self.benchmark_file_name = f"benchmark_{device}.csv"

        # if the file "benchmark.csv" exists, load it to update TIME_TABLE
        if os.path.exists(self.benchmark_file_name):
            self.TIME_TABLE = self.load_time_table_from_csv(self.benchmark_file_name)
        else:
            self.TIME_TABLE = {}
        
        if required_keys is not None:
            self.TIME_TABLE = {k: v for k, v in self.TIME_TABLE.items() if k in required_keys}
        
        self.AVERAGE_inference_time = np.ceil(np.mean(np.array(list(self.TIME_TABLE.values())))*self.fps/1000).astype(int) # in frames
        
    def load_time_table_from_csv(self,
        csv_path: Union[str, Path],
        total_col: str = "Total",
        model_col: str = "Model",
        size_col: str = "InputSize",
        round_to: Optional[int] = 2,
    ) -> Dict[str, float]:

        csv_path = Path(csv_path)

        df = pd.read_csv(csv_path, skipinitialspace=True)

        # Basic validation
        required = {model_col, size_col, total_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

        # 1. Clean model names in the DataFrame first
        # " yolo11n.pt" -> "yolo11n"
        df['clean_model'] = (
            df[model_col]
            .astype(str)
            .str.strip()
            .str.replace(".pt", "", regex=False)
        )

        # 2. Ensure InputSize is numeric in the DataFrame (crucial for correct sorting 288 < 1120)
        df[size_col] = pd.to_numeric(df[size_col], errors="coerce").fillna(0).astype(int)

        # 3. Create a ranking column for sorting
        # n=0, s=1, m=2, l=3, x=4 (Adjust based on your specific needs)
        model_priority = {
            "yolo11n": 0,
            "yolo11s": 1, 
            "yolo11m": 2, 
            "yolo11l": 3, 
            "yolo11x": 4
        }
        
        # Map the priority. Unknown models get 99 (pushed to the end)
        df['rank'] = df['clean_model'].map(model_priority).fillna(99)

        # 4. SORT THE DATAFRAME: Primary key = Rank, Secondary key = InputSize
        df = df.sort_values(by=['rank', size_col])

        # 5. Extract the sorted data to build the dictionary
        models = df['clean_model']
        sizes = df[size_col]
        totals = pd.to_numeric(df[total_col], errors="raise").astype(float)

        # Build keys (e.g., "yolo11n_288")
        keys = models + "_" + sizes.astype(str)

        out: Dict[str, float] = {}
        
        # Since df is sorted, this loop inserts into the dict in the correct order
        for k, v in zip(keys, totals):
            if round_to is not None:
                v = round(float(v), round_to)
            out[str(k)] = float(v)

        return out


    def get_inference_time(self):
        '''
        outputs the inference time in number of frames for each model (array of shape (12,))
        '''
        return np.ceil(np.array(list(self.TIME_TABLE.values())) * self.fps / 1000.0).astype(int)  # in frames
    

    def update_inference_time(self, moddel_name, input_size, new_time):
        # self.TIME_TABLE[f"{moddel_name}_{input_size}"] = self.TIME_TABLE[f"{moddel_name}_{input_size}"]*0.5 + new_time*0.5
        increase_ratio = new_time / self.TIME_TABLE[f"{moddel_name}_{input_size}"] 
        # multiply all TIME_TABLE values by increase_ratio to adjust for overall speed change
        for key in self.TIME_TABLE.keys():
            self.TIME_TABLE[key] = self.TIME_TABLE[key] * increase_ratio

    def save_benchmark(self, loud=False):
        """
        Updates the 'Total' column of the benchmark CSV file with values from self.TIME_TABLE.
        Other columns are preserved.
        """
        if loud:
            for key, value in self.TIME_TABLE.items():
                print(f"{key}: {value}")

        if not os.path.exists(self.benchmark_file_name):
            print(f"File {self.benchmark_file_name} not found. Cannot save benchmark.")
            return

        # Read the existing CSV
        df = pd.read_csv(self.benchmark_file_name, skipinitialspace=True)
        
        # Column names (matching the loader function defaults)
        model_col = "Model"
        size_col = "InputSize"
        total_col = "Total"

        # 1. Reconstruct the keys for the dataframe rows 
        # (Must match logic in load_time_table_from_csv exactly)
        clean_models = (
            df[model_col]
            .astype(str)
            .str.strip()
            .str.replace(".pt", "", regex=False)
        )
        
        # Handle size column safely
        clean_sizes = pd.to_numeric(df[size_col], errors="coerce").fillna(0).astype(int).astype(str)
        
        # Generate the composite keys for matching
        row_keys = clean_models + "_" + clean_sizes

        # 2. Update the 'Total' column where keys match
        # We iterate through the generated keys and update the dataframe if the key exists in our RAM table
        updated_count = 0
        for idx, key in row_keys.items():
            if key in self.TIME_TABLE:
                # Update the value in the dataframe
                df.at[idx, total_col] = self.TIME_TABLE[key]
                updated_count += 1

        # 3. Write back to CSV
        try:
            df.to_csv(self.benchmark_file_name, index=False)
            if loud:
                print(f"Successfully updated {updated_count} rows in {self.benchmark_file_name}")
        except Exception as e:
            print(f"Failed to save benchmark file: {e}")


class ROMA:
    
    iou_threshold = 0.5
    previous_inferenceTime = 3
    previous_detection = None

    def __init__(self, Detectability, InferenceTime):
 
        self.get_detectability = Detectability
        self.inferenceTime = InferenceTime
        self.inf_in_frames = self.inferenceTime.get_inference_time() # inference time in frames for each model (12,)

    def reset(self):
        self.previous_detection = None
        self.previous_inferenceTime = 3

    def runtime_feedback(self, bboxes_new, new_inferenceTime):
        bboxes_old = self.previous_detection

        N = bboxes_old.shape[0]
        M = bboxes_new.shape[0]
        

        iou_matrix = np.zeros((M, N), dtype=np.float32)

        for i in range(M):
            bbox_new_i = np.repeat(bboxes_new[i:i+1, :], N, axis=0)
            _,iou = GIoU(bbox_new_i, bboxes_old)   # shape (N,)
            iou_matrix[i, :] = iou             
        m_tilde = (iou_matrix >= self.iou_threshold).sum()
        
        AP_t = m_tilde / M  # because M is the psudo ground truth here (papar is correct)

        #Eq. (13)
        #m_bar = N - m_tilde # I think this is typo in the paper, should be M - m_tilde because M is the psudo ground truth here
        m_bar = M - m_tilde
        
        #Eq. (14)
        # u = m_bar / new_inferenceTime # I think this is typo in the paper, should be divided by previous_inferenceTime (the age of previous detection)
        u = m_bar / self.previous_inferenceTime


        max_inference_time = int(np.max(self.inf_in_frames))
        q = np.zeros(max_inference_time)
        beta = np.zeros(max_inference_time)
        AP_i = np.zeros(max_inference_time)

        q[0] = M
        beta[0] = 1
        AP_i[0] = AP_t

        for i in range(1, max_inference_time):
            q[i] = q[i-1] - u                  if (q[i-1] - u) > 0 else 0   # Eq. (15)
            q_ratio = q[i] / q[i-1]            if q[i-1] > 0 else 0
            beta[i] = beta[i-1] * q_ratio**2
            AP_i[i] = AP_i[i-1] * beta[i] 

        # return AP_i, beta, u, q

        RAP = np.ones(len(self.inf_in_frames))
        gamma_DEN = np.sum(beta[:new_inferenceTime]) / new_inferenceTime
        
        areas_new = bboxes_new[:,2] * bboxes_new[:,3]  # including not matched boxes
        model = f'{self.current_model_name}_{self.current_inputsize}'
        Detectability = self.get_detectability(model, areas_new) # Eq. (6)
        alpha = Detectability / (M+0.1) # Eq. (18)

        for i, BLOCK_SIZE_M2 in enumerate(self.inf_in_frames):
           
            gamma_NUM = np.sum(beta[:BLOCK_SIZE_M2]) / BLOCK_SIZE_M2
            gamma = gamma_NUM / gamma_DEN
            
            RAP[i] = alpha[i] * gamma # Eq. (17)

        return RAP
    
    def get_feedback(self, pred, inference_time_ms, current_model_name, current_inputsize):

        self.inferenceTime.update_inference_time(current_model_name, current_inputsize, inference_time_ms)

        inference_time_F = np.ceil(inference_time_ms * self.inferenceTime.fps / 1000.0).astype(int)  # in frames
        bboxes_new = xyxy2xywh(pred[:, :4]).cpu().numpy()
        self.current_model_name = current_model_name
        self.current_inputsize = current_inputsize
        self.inf_in_frames = self.inferenceTime.get_inference_time()  # inference time in frames for each model (12,)

        RAP = None
        if self.previous_detection is not None and pred is not None:
            RAP = self.runtime_feedback( #                  <-- Feedback
                bboxes_new,
                new_inferenceTime= inference_time_F
            )

        self.previous_detection = bboxes_new.copy() # store for next cycle
        self.previous_inferenceTime = inference_time_F

        return np.argmax(RAP) if RAP is not None else None



def load_time_table_from_csv(
    csv_path: Union[str, Path],
    total_col: str = "Total",
    model_col: str = "Model",
    size_col: str = "InputSize",
    round_to: Optional[int] = 2,
) -> Dict[str, float]:

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path, skipinitialspace=True)

    # Basic validation
    required = {model_col, size_col, total_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    # 1. Clean model names in the DataFrame first
    # " yolo11n.pt" -> "yolo11n"
    df['clean_model'] = (
        df[model_col]
        .astype(str)
        .str.strip()
        .str.replace(".pt", "", regex=False)
    )

    # 2. Ensure InputSize is numeric in the DataFrame (crucial for correct sorting 288 < 1120)
    df[size_col] = pd.to_numeric(df[size_col], errors="coerce").fillna(0).astype(int)

    # 3. Create a ranking column for sorting
    # n=0, s=1, m=2, l=3, x=4 (Adjust based on your specific needs)
    model_priority = {
        "yolo11n": 0,
        "yolo11s": 1, 
        "yolo11m": 2, 
        "yolo11l": 3, 
        "yolo11x": 4
    }
    
    # Map the priority. Unknown models get 99 (pushed to the end)
    df['rank'] = df['clean_model'].map(model_priority).fillna(99)

    # 4. SORT THE DATAFRAME: Primary key = Rank, Secondary key = InputSize
    df = df.sort_values(by=['rank', size_col])

    # 5. Extract the sorted data to build the dictionary
    models = df['clean_model']
    sizes = df[size_col]
    totals = pd.to_numeric(df[total_col], errors="raise").astype(float)

    # Build keys (e.g., "yolo11n_288")
    keys = models + "_" + sizes.astype(str)

    out: Dict[str, float] = {}
    
    # Since df is sorted, this loop inserts into the dict in the correct order
    for k, v in zip(keys, totals):
        if round_to is not None:
            v = round(float(v), round_to)
        out[str(k)] = float(v)

    return out
