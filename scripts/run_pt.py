
import os
import json
from time import time

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F


class Tester():

    def __init__(self, load_model):

        self.model = torch.jit.load(load_model)
        self.model.eval()
        self.model.cuda()

        for _ in range(10):
            dummy_input = torch.rand(1, 3, 768, 768).cuda()
            self.model(dummy_input)
        print('Warm-up Done !!!')

        self.label_list = ['tigan', 'peitu']

    def _encode_image(self, image_path, input_size=[768, 768]):

        image = cv2.imread(image_path)
        longer_edge, shorter_edge = np.max(image.shape[:2]), np.min(image.shape[:2])
        scale = max(longer_edge / np.max(input_size), shorter_edge / np.min(input_size))
        image = cv2.resize(image, 
            (round(image.shape[1] / scale), round(image.shape[0] / scale)))
        
        h, w, _ = image.shape
        if h > w:
            pad_h, pad_w = np.max(input_size) - h, np.min(input_size) - w
        else:
            pad_h, pad_w = np.min(input_size) - h, np.max(input_size) - w
        image = np.pad(image, 
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)))

        meta = {'scale': scale, 'pad_h': pad_h // 2, 'pad_w': pad_w // 2}
        
        image = image.astype(np.float32) / 255.
        image -= np.array([0.40789654, 0.44719302, 0.47026115]).reshape(1, 1, 3)
        image /= np.array([0.28863828, 0.27408164, 0.27809835]).reshape(1, 1, 3)

        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.float().cuda()
        
        return image_tensor, meta

    def _decode_output(self, bboxes, seg_feat, conv_weights, meta, thresh=0.4):

        _, H, W = seg_feat.size()

        # get boxes (output scale)
        boxs = bboxes.detach().cpu().numpy()
        boxs[:, 0] = np.maximum(boxs[:, 0], 0)
        boxs[:, 1] = np.maximum(boxs[:, 1], 0)
        boxs[:, 2] = np.minimum(boxs[:, 2], W)
        boxs[:, 3] = np.minimum(boxs[:, 3], H)
        k = int(np.sum(boxs[:, 4] > thresh))
        if k == 0: return [], [], [], []

        # get bboxs and scores
        scores = boxs[:k, 4] # k
        if boxs.shape[1] == 6: labels = boxs[:k, 5] # k
        else: labels = np.zeros(k)
        offset = np.array([[meta['pad_w'], meta['pad_h'], meta['pad_w'], meta['pad_h']]])
        boxs_ori = (boxs[:k, :4] * 4 - offset) * meta['scale'] # k x 4

        # position embedding
        x_range = torch.arange(W).float().cuda()
        y_range = torch.arange(H).float().cuda()
        y_grid, x_grid = torch.meshgrid([y_range, x_range]) # H x W
        xs = (bboxes[:k, 0] + bboxes[:k, 2]) / 2. # k
        ys = (bboxes[:k, 1] + bboxes[:k, 3]) / 2. # k
        y_rel_coord = (y_grid.unsqueeze(0) - ys.unsqueeze(1).unsqueeze(1)).unsqueeze(1) / 128.
        x_rel_coord = (x_grid.unsqueeze(0) - xs.unsqueeze(1).unsqueeze(1)).unsqueeze(1) / 128.
        seg_feat = seg_feat.unsqueeze(0).expand(k, 8, H, W)
        seg_feat = torch.cat([seg_feat, x_rel_coord, y_rel_coord], 1).view(1, k * 10, H, W)

        # conditional convolutions
        conv1w, conv1b, conv2w, conv2b, conv3w, conv3b = \
            torch.split(conv_weights[:k, :], [10 * 8, 8, 8 * 8, 8, 8, 1], 1)
        conv1w = conv1w.contiguous().view(k * 8, 10, 1, 1)
        conv1b = conv1b.contiguous().flatten()
        seg_feat = F.conv2d(seg_feat, conv1w, conv1b, groups=k).relu()
        conv2w = conv2w.contiguous().view(k * 8, 8, 1, 1)
        conv2b = conv2b.contiguous().flatten()
        seg_feat = F.conv2d(seg_feat, conv2w, conv2b, groups=k).relu()
        conv3w = conv3w.contiguous().view(k, 8, 1, 1)
        conv3b = conv3b.contiguous().flatten()
        seg_feat = F.conv2d(seg_feat, conv3w, conv3b, groups=k).sigmoid()
        seg_feat = seg_feat.view(k, H, W)
        seg_feat = F.interpolate(seg_feat.view(1, k, H, W), 
            scale_factor=4, mode='bilinear', align_corners=True).squeeze(0)
        seg_masks = seg_feat.detach().cpu().numpy()

        # fit polygons
        segs = []
        for seg_mask, box, label in zip(seg_masks, boxs, labels):
            seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(seg_mask, 
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if label == 0:
                epsilon = (box[2] - box[0]) / 15.
            else:
            	epsilon = min(box[2] - box[0], box[3] - box[1]) / 10.
            contours = [cv2.approxPolyDP(contour, epsilon, True) for contour in contours]
            contours = [[[
                round((p[0][0] - meta['pad_w']) * meta['scale']), 
                round((p[0][1] - meta['pad_h']) * meta['scale'])] \
                for p in contour] \
                for contour in contours]
            segs.append(contours)

        return labels, scores, boxs_ori, segs

    def _visualize(self, image_path, labels, scores, boxs, segs):

        image = cv2.imread(image_path)
        masks = [image.copy() for _ in self.label_list]

        for label, score, box, seg in zip(labels, scores, boxs, segs):
            color = tuple(np.random.rand(3) * 160)
            cv2.rectangle(image, 
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])), color, 2)
            cv2.putText(image, 
                '%s %.3f' %(self.label_list[int(label)], score), 
                (int(box[0]), int(box[1]) - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            for s in seg:
                cv2.fillPoly(masks[int(label)], np.array([s]), color)

        image_vis = image * .5 + np.min(masks, 0) * .5
        # image_vis = image * .5 + masks[1] * .5
        return image_vis

    def process(self, image_dir, output_dir, input_size=[768, 768], vis=True):

        pre_t, infer_t, post_t = 0, 0, 0

        files = [file for file in os.listdir(image_dir) if not file.startswith('.')]
        for file in tqdm(files):

            # if not file.startswith('domestic_'): continue
            # if file != 'ytj_56b4f8f65d310fabf596697a0d749eb9_453B0ED6E13C4938BB37C2FC05DA3848.jpg': continue

            torch.cuda.synchronize()
            s_time = time()
            image_tensor, meta = self._encode_image(
                os.path.join(image_dir, file), input_size)
            torch.cuda.synchronize()
            pre_t += time() - s_time

            torch.cuda.synchronize()
            s_time = time()
            bboxes, seg_feat, conv_weights = self.model(image_tensor)
            torch.cuda.synchronize()
            infer_t += time() - s_time

            torch.cuda.synchronize()
            s_time = time()
            labels, scores, boxs, segs = self._decode_output(
                bboxes[0], seg_feat[0], conv_weights[0], meta)
            torch.cuda.synchronize()
            post_t += time() - s_time

            res_dict = [{
                'bbox' : [float(b) for b in box], 
                'score': float(score),
                'mask' : [[[float(p[0]), float(p[1])] for p in contour] for contour in seg], 
                'class': self.label_list[int(label)]} \
                for label, box, score, seg in zip(labels, boxs, scores, segs)]
            json.dump(res_dict, 
                open(os.path.join(output_dir, 'res', file.replace('.jpg', '.json')), 'w'), 
                indent=4)

            if not vis: continue
            image_vis = self._visualize(
                os.path.join(image_dir, file), labels, scores, boxs, segs)
            cv2.imwrite(os.path.join(output_dir, 'vis', file), image_vis)

        print(pre_t, infer_t, post_t)


if __name__ == '__main__':

    load_model = 'exp/ctseg/qieti_2cls_0511/model_last.pt'
    tester = Tester(load_model)

    image_dir = '/ssd2/shared/data/zypg_data/qieti_tigan/all/photo_qieti_zhiyun/val'
    output_dir = '../qieti_tigan_eval/results/CenterNet-CondInst_0511'
    # image_dir = '/ssd2/exec/liyx/zypg/data/qieti_xb/hw_and_badcase/images'
    # output_dir = '/ssd2/exec/liyx/zypg/data/qieti_xb/hw_and_badcase/preds'
    tester.process(image_dir, output_dir, input_size=[576, 768], vis=False)