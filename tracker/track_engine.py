import cv2
import os
import os.path as osp
import pandas as pd
import numpy as np
from tools.utils import get_image_list
from tools.visualize import plot_tracking
from tools.timer import Timer
from tracker.sri_track.Kfree_tracker_main import Tracker

def image_demo(detector, reid, vis_folder, img_sizes, cfg, d_path, det_path, logger):
    if osp.isdir(d_path):
        files = get_image_list(d_path)
    else:
        files = d_path
    files.sort()

    height = img_sizes[0]
    width = img_sizes[1]
    logger.info(f"ORI IMAGE HEIGHT: {height}")
    logger.info(f"ORI IMAGE WIDTH: {width}")

    tracker = Tracker(cfg, img_sizes, frame_rate=cfg.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        timer.tic()
        
        if frame_id % 30 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        
        frame = cv2.imread(img_path)
        if det_path:
            det_file = open(det_path,'r')
            outputs = []

            for d in det_file.readlines():
                dd = []
                ds = d.split(",")
                x1 = int(ds[2])
                y1 = int(ds[3])
                x2 = x1 + int(ds[4])
                y2 = y1 + int(ds[5])
                score = 0.9
                N1 = 0.9
                N2 = 0.0

                if int(ds[0]) == int(frame_id):
                    dd.append(x1)
                    dd.append(y1)
                    dd.append(x2)
                    dd.append(y2)
                    dd.append(score)
                    dd.append(N1)
                    dd.append(N2)
                    outputs.append(dd)

            det = np.array(outputs)
            cropped_imgs = [frame[max(0,int(y1)):min(height,int(y2)),max(0,int(x1)):min(width,int(x2))] for x1,y1,x2,y2,_,_,_ in det]

        else:    
            outputs, _ = detector.inference(img_path, timer)
            if outputs[0] != None:
                det = outputs[0].cpu().detach().numpy()
                scale = min(1440/width, 800/height)
                det /= scale
                cropped_imgs = [frame[max(0,int(y1)):min(height,int(y2)),max(0,int(x1)):min(width,int(x2))] for x1,y1,x2,y2,_,_,_ in det]
            
        if det is not None:
            online_tlwhs = []
            online_ids = []
            online_scores = []
            
            if cfg.with_reid:
                embs = reid(cropped_imgs)
                embs = embs.cpu().detach().numpy()
            else:
                embs = []

            online_targets, Fshift, svar = tracker.update(det, embs, frame, reid)

            for t in online_targets:

                if cfg.TRACKER == "MY":
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    score = t.score

                else:
                    x1, y1, x2, y2, tid, score = t[:6]
                    tlwh = [x1, y1, x2 - x1, y2 - y1]

                if tlwh[2] * tlwh[3] > cfg.det_min_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n"
                    )
                    
            timer.toc()

            online_im = plot_tracking(
                svar, Fshift, frame, online_tlwhs, online_ids, online_scores, frame_id=frame_id + 1, fps=1. / timer.average_time
            )

        else:
            timer.toc()
            online_im = frame

        if cfg.save_image:
            os.makedirs(vis_folder, exist_ok=True)
            cv2.imwrite(osp.join(vis_folder, osp.basename(img_path)), online_im)
    
        frame_id += 1
    
    print("TOTAL FRAMES: ", frame_id-1)

    if cfg.save_result:
        save_name = os.path.basename(os.path.dirname(d_path)) + '.txt'
        res_file = osp.join(os.path.dirname(vis_folder), save_name)
        with open(res_file, 'w') as f:
            f.writelines(results)