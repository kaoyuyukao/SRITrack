import numpy as np

from tracker.sri_track import matching as matching
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .cmc import get_cmc_method
from .strack import STrack

class Tracker(object):
    def __init__(self, cfg, img_sizes, frame_rate=30):

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        BaseTrack.clear_count()

        self.frame_id = 0
        self.cfg = cfg
        self.img_sizes = img_sizes

        self.track_high_th = cfg.track_high_th
        self.track_low_th = cfg.track_low_th
        self.new_track_th = cfg.track_new_th

        self.buffer_size = int(frame_rate / 30.0 * cfg.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # My tracker
        self.track_p_th = cfg.track_p_th
        self.track_v_th = cfg.track_vc_th
        self.track_vc_th = cfg.track_vc_th
        self.track_vf_th = cfg.track_vf_th
        self.cmc = get_cmc_method('sof')()
        self.track_b_th = cfg.track_b_th
        self.track_new_th = cfg.track_new_th


    def update(self, output_results, embedding, frame, reid):
    
        '''
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        '''
        
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
    
        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:,4]
                bboxes = output_results[:, :4]  # x1y1x2y2
                x1 = output_results[:,0]
                x2 = output_results[:,2]
                y1 = output_results[:,1]
                y2 = output_results[:,3]
            elif output_results.shape[1] == 7:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                x1 = output_results[:,0]
                x2 = output_results[:,2]
                y1 = output_results[:,1]
                y2 = output_results[:,3]
            else:
                raise ValueError('Wrong detection size {}'.format(output_results.shape[1]))            


            w = output_results[:,2] - output_results[:,0]
            w_m = np.median(w, axis=0)
            h = output_results[:,3] - output_results[:,1]
            h_m = np.median(h, axis=0)

            if w_m < h_m:
                s_var = w_m * self.track_b_th
            else:
                s_var = h_m * self.track_b_th

            # Boundary detect for width
            l_b = int(s_var)
            r_b = self.img_sizes[1] - int(s_var)
            
            #Remove boundary detections
            b_inds_l = x2 > l_b
            b_inds_r = x1 < r_b

            # Boundary detect for height
            t_b = int(s_var)
            b_b = self.img_sizes[0] - int(s_var)
            
            #Remove boundary detections
            b_inds_t = y2 > t_b
            b_inds_b = y1 < b_b

            
            # Remove bad detections
            lowest_inds = scores > self.track_low_th 
            
            if self.cfg.ris:
                det_mask = (lowest_inds) & (b_inds_r) & (b_inds_l) & (b_inds_t) & (b_inds_b)
            else:
                det_mask = lowest_inds

            bboxes = bboxes[det_mask]
            scores = scores[det_mask]
            occs = compute_det_occlusion(bboxes)
            
            # Find high threshold detections
            remain_inds = scores > self.track_high_th
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            occs_keep = occs[remain_inds]
            
            if self.cfg.with_reid:
                embedding = embedding[det_mask]
                features_keep = embedding[remain_inds]

        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []
            features_keep = []

        if len(dets) > 0:
            '''Detections'''
            if self.cfg.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, feat=f, role = 'det') for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Associate with high score detection boxes
        transform = self.cmc.apply(frame, dets)
        STrack.multi_gmc(strack_pool, transform)
        STrack.multi_gmc(unconfirmed, transform)
        Fshift = transform[:2, 2] 

        if self.cfg.EIoU:
            init_expand_scale = 0.7    
            expand_scale_step = 0.1
            cur_expand_scale = init_expand_scale + expand_scale_step

            ious_dists = matching.eiou_distance(strack_pool, detections, cur_expand_scale, self.frame_id, 0.0)
            ious_dists_mask = (ious_dists > self.track_p_th)

            if self.cfg.with_reid:
                if self.cfg.vp_dga:                    
                    emb_dists_r = matching.embedding_distance(strack_pool, detections, self.frame_id, 2.0) / 2.0
                    emb_dists_r[emb_dists_r >= self.track_vc_th] = 1.0
                                    
                    emb_dists = matching.embedding_distance(strack_pool, detections, self.frame_id, 2.0) / 2.0
                    emb_dists[emb_dists > self.track_vf_th] = 1.0
                    emb_dists[ious_dists_mask] = 1.0
                    
                    dists = np.minimum(ious_dists, emb_dists)
                    dists = np.maximum(dists, emb_dists_r)

                else:               
                    emb_dists = matching.embedding_distance(strack_pool, detections, self.frame_id, 2.0) / 2.0
                    emb_dists[emb_dists > self.track_vf_th] = 1.0
                    emb_dists[ious_dists_mask] = 1.0
                    
                    dists = np.minimum(ious_dists, emb_dists)

            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.cfg.track_match_th)
        
        else:
            
            ious_dists = matching.kfree_iou_distance(strack_pool, detections, self.frame_id, 0.0)
            ious_dists_mask = (ious_dists > self.track_p_th)

            if self.cfg.with_reid:
                if self.cfg.vp_dga:                    
                    emb_dists_r = matching.embedding_distance(strack_pool, detections, self.frame_id, 2.0) / 2.0
                    emb_dists_r[emb_dists_r >= self.track_vc_th] = 1.0
                                    
                    emb_dists = matching.embedding_distance(strack_pool, detections, self.frame_id, 2.0) / 2.0
                    emb_dists[emb_dists > self.track_vf_th] = 1.0
                    emb_dists[ious_dists_mask] = 1.0
                    
                    dists = np.minimum(ious_dists, emb_dists)
                    dists = np.maximum(dists, emb_dists_r)

                else:               
                    emb_dists = matching.embedding_distance(strack_pool, detections, self.frame_id, 2.0) / 2.0
                    emb_dists[emb_dists > self.track_vf_th] = 1.0
                    emb_dists[ious_dists_mask] = 1.0
                    
                    dists = np.minimum(ious_dists, emb_dists)

            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.cfg.track_match_th)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                
                if occs_keep[idet] < 1.0:
                    track.update_h(detections[idet], self.frame_id)
                else:
                    track.update_l(detections[idet], self.frame_id) # update_l

                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        strack_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        detections = [detections[i] for i in u_detection]

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.cfg.track_high_th
            inds_low = scores > self.cfg.track_low_th
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            if self.cfg.with_reid:
                features_second = embedding[inds_second]
        else:
            dets_second = []
            scores_second = []
            features_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if self.cfg.with_reid:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, feat=f, role='det') for
                                    (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                    (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        
        if self.cfg.EIoU:
            dists = matching.eiou_distance(r_tracked_stracks, detections_second, 0.5, self.frame_id, 0.0)
        else:
            dists = matching.kfree_iou_distance(r_tracked_stracks, detections_second, self.frame_id, 0.0)

        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update_l(det, self.frame_id) # update_l
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        if self.cfg.EIoU:
            ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5, self.frame_id, 0.0)
        else:
            ious_dists = matching.kfree_iou_distance(unconfirmed, detections, self.frame_id, 0.0)

        ious_dists_mask = (ious_dists > self.track_p_th)

        if self.cfg.with_reid:
            
            if self.cfg.vp_dga:
                emb_dists_r = matching.embedding_distance(unconfirmed, detections, self.frame_id, 2.0) / 2.0
                emb_dists_r[emb_dists_r >= self.track_vc_th] = 1.0
                emb_dists = matching.embedding_distance(unconfirmed, detections, self.frame_id, 2.0) / 2.0
                emb_dists[emb_dists > self.track_vf_th] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
                dists = np.maximum(dists, emb_dists_r)
            
            else:
                emb_dists = matching.embedding_distance(unconfirmed, detections, self.frame_id, 2.0) / 2.0
                emb_dists[emb_dists > self.track_p_th] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
                
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update_h(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_new_th:
                continue

            track.activate(self.kalman_filter, self.frame_id, detections[inew].curr_feat)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks, frame_id=self.frame_id)
        output_stracks = [track for track in self.tracked_stracks]

        return output_stracks, Fshift, s_var

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb, frame_id=None):
    pdist = matching.iou_distance(stracksa, stracksb, current_frame=frame_id, lambda_time=0.0)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def _union_area(rects):
    if not rects:
        return 0.0
    events = []
    for x1,y1,x2,y2 in rects:
        events.append((x1, 1, y1, y2))
        events.append((x2,-1, y1, y2))
    events.sort()
    active = []
    area = 0.0
    prev_x = events[0][0]

    def y_covered(segs):
        if not segs: return 0.0
        segs = sorted(segs)
        total, cur_y1, cur_y2 = 0.0, *segs[0]
        for y1,y2 in segs[1:]:
            if y1 <= cur_y2:
                cur_y2 = max(cur_y2, y2)
            else:
                total += cur_y2 - cur_y1
                cur_y1, cur_y2 = y1, y2
        return total + (cur_y2 - cur_y1)

    for x,typ,y1,y2 in events:
        dx = x - prev_x
        if dx>0:
            area += dx * y_covered(active)
            prev_x = x
        if typ==1:
            active.append((y1,y2))
        else:
            active.remove((y1,y2))
    return area

def compute_det_occlusion(bboxes):
    N = len(bboxes)
    if N==0:
        return np.array([])
    occ = np.zeros(N, dtype=np.float32)
    x1,y1,x2,y2 = bboxes.T
    w = x2-x1
    h = y2-y1
    areas = w*h

    for i in range(N):
        rects=[]
        for j in range(N):
            if i==j: continue
            ix1 = max(x1[i],x1[j])
            iy1 = max(y1[i],y1[j])
            ix2 = min(x2[i],x2[j])
            iy2 = min(y2[i],y2[j])
            if ix2>ix1 and iy2>iy1:
                rects.append((ix1,iy1,ix2,iy2))
        inter_union = _union_area(rects)
        occ[i] = min(inter_union / (areas[i]+1e-6), 1.0)
    return occ
