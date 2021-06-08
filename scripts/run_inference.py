
import sys
sys.path.append('src')
import _init_paths

import os
import cv2
import json
from tqdm import tqdm
from time import time
import numpy as np
import torch

from pycocotools import mask as mask_utils


class Tester():

    def __init__(self, arch, load_model):

        from opts import opts
        opt = opts().init()
        opt.test = True
        opt.arch = arch
        if arch.startswith('dla'): opt.head_conv = 256
        opt.num_classes = 1
        opt.heads = {
            'hm': opt.num_classes, 'wh': 2, 'reg': 2,
            'conv_weight': 169, 'seg_feat': 8}
        opt.load_model = load_model
        opt.input_h, opt.input_w = 768, 768
        self.opt = opt

        from detectors.detector_factory import detector_factory
        self.detector = detector_factory[opt.task](opt)

        self.class_list = ['tigan']

        for _ in range(10):
            dummy_input = torch.rand(1, 3, 768, 768).cuda()
            self.detector.model(dummy_input)
        print('Warm-up Done !!!')

    def infer(self, file_dir, out_dir):

        infer_t, post_t = 0, 0

        files = os.listdir(file_dir)
        for file in tqdm(files):
            
            if file.startswith('.'): continue
            
            image_path = os.path.join(file_dir, file)
            image = cv2.imread(image_path)
            seg_mask = image.copy()

            ret = self.detector.run(image_path)
            infer_t += ret['net']
            post_t += ret['post']
            
            res_dict = []
            for cls_id, cls_name in zip(range(1, self.opt.num_classes + 1), self.class_list):
                ret_res_cls = ret['results'][cls_id]
                for box, mask in zip(ret_res_cls['boxs'], ret_res_cls['pred_mask']):
                    
                    score = float(box[4])
                    if score < 0.1: continue

                    bbox = [
                        float(max(box[0], 0)), float(max(box[1], 0)),
                        float(min(box[2], image.shape[1])), float(min(box[3], image.shape[0]))]
                    if box[3] - box[1] <= 0 or box[2] - box[0] <= 0: continue
                    res_dict.append({
                        'bbox' : bbox,
                        'score': score,
                        'class': cls_name})

                    color = tuple(np.random.rand(3) * 160)
                    cv2.rectangle(image, 
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.putText(image, 
                        '%.3f' %score, (int(bbox[0]), int(bbox[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    mask = mask_utils.decode(mask)
                    seg_mask[mask == 1, :] = color

            cv2.imwrite(os.path.join(out_dir, 'vis', file), image * .5 + seg_mask * .5)
            json.dump(res_dict, 
                open(os.path.join(out_dir, 'res', file.replace('.jpg', '.json')), 'w'), indent=4)

        print(infer_t, post_t)

    def trace(self, output_path):

        dummy_input = torch.rand(1, 3, 512, 512).cuda()
        dummy_output = self.detector.model(dummy_input)
        
        traced_module = torch.jit.trace(self.detector.model, dummy_input)
        traced_module.save(output_path)


if __name__ == '__main__':

    arch = 'dlav0_34'
    load_model = 'exp/ctseg/qieti_peitu/model_last.pth'
    tester = Tester(arch, load_model)

    # output_path = 'exp/ctseg/qieti_all_dla34_512/model_last.pt'
    # tester.trace(output_path)

    # file_dir = '/ssd2/shared/data/zypg_data/qieti_tigan/all/photo_qieti/val'
    file_dir = '/ssd2/shared/data/zypg_data/qieti_tigan/all/photo_qieti/images'
    out_dir = '../qieti_tigan_eval/results/peitu'
    tester.infer(file_dir, out_dir)
