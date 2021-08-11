
import sys
sys.path.append('../AdelaiDet/scripts')
from evaluate_multi_class import evaluate

import json
import numpy as np
from matplotlib import pyplot as plt


class Opt():
    def __init__(self, pred_dir, confidence):
        self.pred_dir = pred_dir
        self.confidence = confidence
        self.iou_threshs = '0.5'
        self.log_file = 'None'


def evaluate_all(pred_dir, conf_list):

    F_list = []
    for conf in conf_list:
        opt = Opt(pred_dir, conf)
        res_dict = evaluate(opt)
        F_list.append(res_dict[0]['det_evals']['F1_measure'])

    return F_list


def plot_curves(F_list):

    F_list_cls = []
    for i in range(len(F_list[0])):
        F_list_cls.append([Fs[i] for Fs in F_list])

    for Fs in F_list_cls:
        plt.plot(conf_list, Fs)
        print(conf_list[np.argmax(Fs)], Fs[np.argmax(Fs)])
    plt.legend(['Formula', 'Figure', 'Answer_XHX', 'Answer_KH', 'DTH', 'XTH', 'Extra_Info'])
    plt.xlabel('Threshold for confidence')
    # plt.savefig('fuck.jpg', dpi=400)


if __name__ == '__main__':

    pred_dir = 'exp/ctdet/7cls_hg2/res_768'
    conf_list = [.05 * i for i in range(19)]
    F_list = evaluate_all(pred_dir, conf_list)
    plot_curves(F_list)
    
