
import sys
sys.path.append('src')
import _init_paths

import os
import cv2
import json
from tqdm import tqdm


class Tester():

    def __init__(self, arch, load_model):

        from opts import opts
        opt = opts().init()
        opt.test = True
        opt.arch = arch
        opt.num_classes = 3
        opt.heads = {'hm': opt.num_classes, 'wh': 2, 'reg': 2}
        opt.load_model = load_model
        opt.input_h, opt.input_w = 768, 768
        self.opt = opt

        from detectors.detector_factory import detector_factory
        self.detector = detector_factory[opt.task](opt)

        self.class_list = ['timu', 'dati', 'peitu']

    def infer(self, file_dir, out_dir):

        files = os.listdir(file_dir)
        for file in tqdm(files):
            
            if file.startswith('.'): continue
            
            image_path = os.path.join(file_dir, file)
            image = cv2.imread(image_path)
            ret = self.detector.run(image_path)
            
            res_dict = []
            for cls_id, cls_name in zip(range(1, self.opt.num_classes + 1), self.class_list):
                for inst in ret['results'][cls_id]:
                    bbox = [
                        float(max(inst[0], 0)), float(max(inst[1], 0)),
                        float(min(inst[2], image.shape[1])), float(min(inst[3], image.shape[0]))]
                    if bbox[3] - bbox[1] <= 0 or bbox[2] - bbox[0] <= 0: continue
                    res_dict.append({
                        'bbox' : bbox,
                        'score': float(inst[4]),
                        'class': cls_name})
            json.dump(res_dict, 
                open(os.path.join(out_dir, file.replace('.jpg', '.json')), 'w'), indent=4)


if __name__ == '__main__':

    arch = 'res_101'
    load_model = 'models/model_qieti_shiqi.pth'
    tester = Tester(arch, load_model)

    file_dir = '/ssd2/shared/data/zypg_data/qieti_tigan/zyb/images'
    out_dir = 'temp_res'
    tester.infer(file_dir, out_dir)
