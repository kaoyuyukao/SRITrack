import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
from tracker.sri_track import kalman_filter

def _iou(boxes1, boxes2):
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    x1 = np.maximum(boxes1[:, np.newaxis, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, np.newaxis, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, np.newaxis, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, np.newaxis, 3], boxes2[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = boxes1_area[:, np.newaxis] + boxes2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def ious(atlbrs, btlbrs):
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = _iou(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )
    return ious


def expand(tlbr, e):
    
    t,l,b,r = tlbr
    w = r-l
    h = b-t
    expand_w = 2*w*e + w
    expand_h = 2*h*e + h

    new_tlbr = [t-expand_h//2,l-expand_w//2,b+expand_h//2,r+expand_w//2]

    return new_tlbr

def eious(atlbrs, btlbrs, e):
    """
    Compute cost based on EIoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    eious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if eious.size == 0:
        return eious

    atlbrs = np.array([expand(tlbr, e) for tlbr in atlbrs])
    btlbrs = np.array([expand(tlbr, e) for tlbr in btlbrs])

    eious = _iou(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return eious

def iou_distance(atracks, btracks, current_frame, lambda_time):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    base_cost = 1 - _ious

    disappear_penalties = []
    for track in atracks:
        disappear_frames = current_frame - track.end_frame - 1
        penalty = (disappear_frames/current_frame) * lambda_time # 0.0-1.0
        disappear_penalties.append(penalty)
        
    
    disappear_penalties = np.array(disappear_penalties)
    penalty_matrix = disappear_penalties[:, np.newaxis]
    cost_matrix = base_cost * (1 + penalty_matrix)

    return cost_matrix

def kfree_iou_distance(atracks, btracks, current_frame, lambda_time):

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    base_cost = 1 - _ious

    disappear_penalties = []
    for track in atracks:
        disappear_frames = current_frame - track.end_frame - 1
        penalty = (disappear_frames/current_frame) * lambda_time # 0.0-1.0
        disappear_penalties.append(penalty)
    
    disappear_penalties = np.array(disappear_penalties)
    penalty_matrix = disappear_penalties[:, np.newaxis]
    cost_matrix = base_cost * (1 + penalty_matrix)

    return cost_matrix

def kalman_eiou_distance(atracks, btracks, expand, current_frame, lambda_time):
    """
    Hsiang-Wei Huang; Cheng-Yen Yang; Jiacheng Sun; Pyong-Kun Kim; Kwang-Ju Kim; Kyoungoh Lee
    "Iterative Scale-Up ExpansionIoU and Deep Features Association for Multi-Object Tracking in Sports," 
    2024 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW),
    pp. 163-172, 
    doi: 10.1109/WACVW60836.2024.00024.
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = eious(atlbrs, btlbrs, expand)
    base_cost = 1 - _ious

    disappear_penalties = []
    for track in atracks:
        disappear_frames = current_frame - track.end_frame - 1
        penalty = (disappear_frames/current_frame) * lambda_time # 0.0-1.0
        disappear_penalties.append(penalty)
        
    
    disappear_penalties = np.array(disappear_penalties)
    penalty_matrix = disappear_penalties[:, np.newaxis]
    cost_matrix = base_cost * (1 + penalty_matrix)

    return cost_matrix

def eiou_distance(atracks, btracks, expand, current_frame, lambda_time):

    """
    Hsiang-Wei Huang; Cheng-Yen Yang; Jiacheng Sun; Pyong-Kun Kim; Kwang-Ju Kim; Kyoungoh Lee
    "Iterative Scale-Up ExpansionIoU and Deep Features Association for Multi-Object Tracking in Sports," 
    2024 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW),
    pp. 163-172, 
    doi: 10.1109/WACVW60836.2024.00024.
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.last_tlbr for track in atracks]
        btlbrs = [track.last_tlbr for track in btracks]

    _ious = eious(atlbrs, btlbrs, expand)
    base_cost = 1 - _ious

    disappear_penalties = []
    for track in atracks:
        disappear_frames = current_frame - track.end_frame - 1
        penalty = (disappear_frames/current_frame) * lambda_time # 0.0-1.0
        disappear_penalties.append(penalty)
        
    
    disappear_penalties = np.array(disappear_penalties)
    penalty_matrix = disappear_penalties[:, np.newaxis]
    cost_matrix = base_cost * (1 + penalty_matrix)

    return cost_matrix

def embedding_distance(tracks, detections, current_frame, lambda_time, metric='cosine'):

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)

    if cost_matrix.size == 0:
        return cost_matrix
    
    det_features = np.asarray([det.curr_feat for det in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    base_cost = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    disappear_penalties = []

    for track in tracks:
        disappear_frames = current_frame - track.end_frame - 1
        penalty = (disappear_frames/current_frame) * lambda_time # 0.0-1.0
        disappear_penalties.append(penalty)
    
    disappear_penalties = np.array(disappear_penalties)
    penalty_matrix = disappear_penalties[:, np.newaxis]
    cost_matrix = base_cost * (1 + penalty_matrix)

    return cost_matrix
