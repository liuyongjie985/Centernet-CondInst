
import sys
sys.path.append('src')
import _init_paths

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opts import opts
from detectors.detector_factory import detector_factory


def opt_init(arch, load_model):

    opt = opts().init()

    opt.num_classes = 2
    opt.load_model = load_model
    opt.arch = arch
    if arch.startswith('dla'):
        opt.head_conv = 256

    opt.task = 'ctseg'
    opt.heads = {
        'hm': opt.num_classes, 
        'wh': 2, 
        'reg': 2,
        'conv_weight': 169, 
        'seg_feat': 8}
    
    opt.test = True
    opt.input_h, opt.input_w = 768, 768

    return opt


def encode_image(image_path, input_size=[768, 768]):

    image = cv2.imread(image_path)
    image = cv2.resize(image, tuple(input_size))

    image = image.astype(np.float32) / 255.
    image -= np.array([0.40789654, 0.44719302, 0.47026115]).reshape(1, 1, 3)
    image /= np.array([0.28863828, 0.27408164, 0.27809835]).reshape(1, 1, 3)

    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.float().cuda()
    return image_tensor


class SimpleModel(nn.Module):

    def __init__(self, opt):
        super(SimpleModel, self).__init__()

        self.opt = opt
        self.model = detector_factory[opt.task](opt).model

    def forward(self, x):

        raw_outputs = self.model(x)
        outputs = torch.cat([
            raw_outputs[0]['hm'], 
            raw_outputs[0]['wh'], 
            raw_outputs[0]['reg'], 
            raw_outputs[0]['seg_feat'], 
            raw_outputs[0]['conv_weight']], 1)
        return outputs


class InferModel(nn.Module):

    def __init__(self, opt):
        super(InferModel, self).__init__()

        self.opt = opt
        self.model = detector_factory[opt.task](opt).model

    def _nms(self, hm):
        
        hmax = F.max_pool2d(hm, (3, 3), stride=1, padding=1)
        return hm * (1 - torch.ceil(torch.abs(hmax - hm)))
        # return hm * (hmax == hm).float()

    def _topk(self, hm, K=64):

        B, C, H, W = hm.size()

        topk_vals, topk_inds = torch.topk(hm.view(B, C, H * W), K) # B x C x K
        topk_inds = topk_inds - (topk_inds / H / W).int() * H * W # B x C x K
        topk_ys = (topk_inds / W).int().float() # B x C x K
        topk_xs = (topk_inds - topk_ys * W).float() # B x C x K

        topk_vals, topk_inds_ = torch.topk(topk_vals.view(B, C * K), K) # B x K
        topk_clss = (topk_inds_ / K).int() # B x K
        topk_inds = self._gather_feat_cls(
            topk_inds.view(B, C * K, 1), topk_inds_).view(B, K)
        topk_ys = self._gather_feat_cls(
            topk_ys.view(B, C * K, 1), topk_inds_).view(B, K)
        topk_xs = self._gather_feat_cls(
            topk_xs.view(B, C * K, 1), topk_inds_).view(B, K)
        
        return topk_vals, topk_inds, topk_clss, topk_xs, topk_ys

    def _gather_feat_cls(self, feat, ind):
        
        B, C, W = feat.size()
        B, K = ind.size()
        ind  = ind.unsqueeze(2).expand(B, K, W)
        feat = feat.gather(1, ind)
        return feat

    def _gather_feat_img(self, feat, ind):

        B, K = ind.size()
        B, C, H, W = feat.size()
        ind = ind.unsqueeze(2).expand(B, K, C) # B x K x C
        feat = feat.view(B, C, H * W).permute(0, 2, 1) # B x HW x C
        feat = feat.gather(1, ind) # B x K x C
        return feat

    def _get_boxes(self, topk_xs, topk_ys, topk_vals, topk_clss, reg, wh):
      
        topk_xs = topk_xs + reg[:, :, 0] # B x K
        topk_ys = topk_ys + reg[:, :, 1] # B x K
        bboxes = torch.stack([
            topk_xs - wh[:, :, 0] / 2, topk_ys - wh[:, :, 1] / 2,
            topk_xs + wh[:, :, 0] / 2, topk_ys + wh[:, :, 1] / 2], 2) # B x K x 4
        bboxes = torch.cat([bboxes, topk_vals.unsqueeze(2), 
            topk_clss.float().unsqueeze(2)], 2) # B x K x 6
        return bboxes

    def forward(self, x):

        ## Model inference
        raw_outputs = self.model(x)

        ## Get instances from heat map
        hm = self._nms(raw_outputs[0]['hm'].sigmoid())
        topk_vals, topk_inds, topk_clss, topk_xs, topk_ys = self._topk(hm)

        ## Get bounding boxes
        reg = self._gather_feat_img(raw_outputs[0]['reg'], topk_inds) # B x K x 2
        wh = self._gather_feat_img(raw_outputs[0]['wh'], topk_inds) # B x K x 2
        bboxes = self._get_boxes(topk_xs, topk_ys, topk_vals, topk_clss, reg, wh)

        ## Get masks
        seg_feat = raw_outputs[0]['seg_feat'] # B x 8 x H x W
        conv_weights = self._gather_feat_img(raw_outputs[0]['conv_weight'], topk_inds) # B x K x 169
        
        return bboxes, seg_feat, conv_weights


if __name__ == '__main__':

    arch = 'dlav0_34'
    load_model = 'exp/ctseg/qieti_2cls_zhiyun/model_last.pth'
    opt = opt_init(arch, load_model)
    
    # model = InferModel(opt)
    model = SimpleModel(opt)
    model.eval()
    model.cuda()

    # image_path = 'data/qieti_tigan/val/phone_257314_1.jpg'
    # dummy_input = encode_image(image_path)
    # model(dummy_input)

    # output_path = 'exp/ctseg/qieti_2cls_zhiyun/model_last.pt'
    # traced_module = torch.jit.trace(model, dummy_input)
    # traced_module.save(output_path)

    h, w = 576, 768
    output_path = 'exp/ctseg/qieti_2cls_zhiyun/model_last_%sx%s.onnx' %(h, w)
    dummy_input = torch.rand(1, 3, h, w).cuda()
    torch.onnx.export(
        model, dummy_input, output_path,
        do_constant_folding=True)
