# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-
# # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np
import cv2
import colorsys

trj = {}
__all__ = ["vis"]

def get_color(idx, alpha=1.0):
    hue = (idx * 37) % 360
    rgb = colorsys.hsv_to_rgb(hue / 360, 0.6, 0.9)
    color = tuple(int(c * 255) for c in rgb)
    
    color_with_alpha = tuple(int(c * alpha) for c in color)
    return color_with_alpha

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    global trj  
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = 2
    text_thickness = 2
    bbox_thickness = 2
    max_line_width = 6
    min_line_width = 2
    max_alpha = 0.8 
    min_alpha = 0.2
    alpha = 1 

    cv2.putText(im, 'frame: %d num: %d' % (frame_id - 1, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    if frame_id==0:
        trj = {} #initialize

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        obj_id = int(obj_ids[i])
        center = (int(x1 + w / 2), int(y1 + h / 2)) 

        if obj_id not in trj:
            trj[obj_id] = []
        trj[obj_id].append(center)

        max_length = 30
        if len(trj[obj_id]) > max_length:
            trj[obj_id] = trj[obj_id][-max_length:]

        
        for j in range(1, len(trj[obj_id])):
            overlay = im.copy()
            alpha = min_alpha + (max_alpha - min_alpha) * (j / max_length)
            line_thickness = int(min_line_width + (max_line_width - min_line_width) * (j / max_length))
            t_color = get_color(abs(obj_id), 1)

            cv2.line(overlay, trj[obj_id][j - 1], trj[obj_id][j], t_color, thickness=line_thickness)
            cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        obj_id = int(obj_ids[i])

        bbox_color = get_color(abs(obj_id), 1)
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=bbox_color, thickness=bbox_thickness)

        id_text = '{}'.format(obj_id)
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)[0]
        text_x, text_y = intbox[0], intbox[1] - 5
        bg_x1, bg_y1 = text_x, text_y - text_size[1] - 4
        bg_x2, bg_y2 = text_x + text_size[0] + 6, text_y + 4

        cv2.rectangle(im, (bg_x1, bg_y1), (bg_x2, bg_y2), bbox_color, thickness=-1)
        cv2.putText(im, id_text, (text_x + 2, text_y), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255),
                    thickness=text_thickness)
    return im


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

# def get_color(idx, alpha=1.0):
#     hue = (idx * 37) % 360
#     rgb = colorsys.hsv_to_rgb(hue / 360, 0.5, 0.9 * alpha)
#     color = tuple(int(c * 255) for c in rgb)
#     return color


# def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
#     global trajectories
#     im = np.ascontiguousarray(np.copy(image))
#     im_h, im_w = im.shape[:2]

#     top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

#     #text_scale = max(1, image.shape[1] / 1600.)
#     #text_thickness = 2
#     #line_thickness = max(1, int(image.shape[1] / 500.))
#     text_scale = 2
#     text_thickness = 2
#     line_thickness = 2
#     max_line_width = 6
#     min_line_width = 2
#     max_alpha = 0.8
#     min_alpha = 0.2
#     radius = max(5, int(im_w / 140.))
#     overlay = im.copy()

#     radius = max(5, int(im_w/140.))
#     #cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
#     #            (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
#     cv2.putText(im, 'frame: %d num: %d' % (frame_id-1, len(tlwhs)),
#                 (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

#     for i, tlwh in enumerate(tlwhs):
#         x1, y1, w, h = tlwh
#         obj_id = int(obj_ids[i])

#         center = (int(x1 + w / 2), int(y1 + h / 2))  # 计算目标中心点
#         if obj_id not in trajectories:
#             trajectories[obj_id] = []
#         trajectories[obj_id].append(center)

#         max_length = 30
#         if len(trajectories[obj_id]) > max_length:
#             trajectories[obj_id] = trajectories[obj_id][-max_length:]

#         for j in range(1, len(trajectories[obj_id])):
#             alpha = min_alpha + (max_alpha - min_alpha) * (j / max_length) 
#             t_color = get_color(abs(obj_id), 1)
#             color_bgr = (t_color[2], t_color[1], t_color[0])
#             t_line_thickness = int(min_line_width + (max_line_width - min_line_width) * (j / max_length))
#             cv2.line(overlay, trajectories[obj_id][j - 1], trajectories[obj_id][j], color_bgr, thickness=t_line_thickness)
        
#         cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)

#         alpha = 1
#         b_color = get_color(abs(obj_id), alpha)
#         intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
#         cv2.rectangle(im, intbox[0:2], intbox[2:4], color=b_color, thickness=line_thickness)

#         id_text = '{}'.format(int(obj_id))
#         if ids2 is not None:
#             id_text = id_text + ', {}'.format(int(ids2[i]))
#         cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
#                     thickness=text_thickness)
#     return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
