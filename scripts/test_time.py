
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

    def _encode_image(self, image_path, input_size):

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

    def process(self, image_dir, input_size, batchsize):

        infer_t = 0

        files = os.listdir(image_dir)
        batch_num = len(files) // batchsize

        for idx in tqdm(range(batch_num)):

            image_tensors = []
            for file in files[idx * batchsize : (idx + 1) * batchsize]:
                image_tensor, _ = self._encode_image(
                    os.path.join(image_dir, file), input_size)
                if image_tensor.shape[2] > image_tensor.shape[3]:
                    image_tensor = image_tensor.permute(0, 1, 3, 2)
                image_tensors.append(image_tensor)
            input_tensor = torch.cat(image_tensors, 0)
 
            torch.cuda.synchronize()
            s_time = time()
            self.model(input_tensor)
            torch.cuda.synchronize()
            infer_t += time() - s_time

        print(infer_t / batch_num)


if __name__ == '__main__':

    load_model = 'exp/ctseg/qieti_all_dla34_768/model_last.pt'
    tester = Tester(load_model)

    image_dir = '/ssd2/shared/data/zypg_data/qieti_tigan/all/photo_qieti/val'
    tester.process(image_dir, input_size=[576, 768], batchsize=8)
