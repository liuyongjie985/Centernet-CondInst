
import os
import json
import random
import numpy as np


image_dir = '/ssd2/shared/data/zypg_data/tiku/data_v6/testset_v2/images'
json_dir  = '/ssd2/shared/data/zypg_data/tiku/data_v6/testset_v2/annos'

file_ids = [file.strip() for file in open('data/formula_det7/annotations/test_file.list').readlines()]
file_dict = {file_id:idx for idx, file_id in enumerate(file_ids)}

anno_ids = [file.strip() for file in open('data/formula_det7/annotations/test_anno.list').readlines()]
anno_dict = {anno_id:idx for idx, anno_id in enumerate(anno_ids)}

class_dict = {
    'formula' : 1,
    'peitu'   : 2,
    'xhx'     : 3,
    'kh'      : 4,
    'dth'     : 5,
    'xth'     : 6,
    'twxx'    : 7}


def make_data():

    images = []
    annotations = []

    for file_id in file_ids:

        image_id = file_dict[file_id]
        info_dict = json.load(open(os.path.join(json_dir, file_id + '.json')))
        
        image = {
            "license": 0,
            "file_name": file_id + '.jpg',
            "coco_url": "",
            "height": info_dict['imageHeight'],
            "width": info_dict['imageWidth'],
            "date_captured":"",
            "flickr_url":"",
            "id": image_id}
        images.append(image)

        for idx, shape in enumerate(info_dict['shapes']):
            anno_id = file_id + '_%d' %idx
            segs = [[
                shape['points'][0][0], shape['points'][0][1], 
                shape['points'][1][0], shape['points'][0][1],
                shape['points'][1][0], shape['points'][1][1], 
                shape['points'][0][0], shape['points'][1][1]]]
            bbox = [segs[0][0], segs[0][1], segs[0][4] - segs[0][0], segs[0][5] - segs[0][1]]
            category_id = class_dict[shape['label']] 
            annotation = {
                "id": anno_dict[anno_id],
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segs,
                "area": bbox[2] * bbox[3],
                "bbox": bbox,
                "iscrowd": 0}
            annotations.append(annotation)

    return images, annotations


if __name__ == '__main__':

    info_dict = {}

    info_dict['info'] = {
        "description": "Formula detection for Zhiyun-Tiku (7 classes)",
        "url": "",
        "version": "1.0",
        "year": 2020,
        "contributor": "Li Yixin",
        "date_created": "2021/01/18"}

    info_dict['licenses'] = [
        {
            "url": "",
            "id" : 0,
            "name": "Do not distribute under any fucking status !!!"
        }]
    
    info_dict['categories'] = [
        {"supercategory": "none", "id": idx, "name": name} \
        for name, idx in class_dict.items()]

    info_dict['images'], info_dict['annotations'] = make_data()

    output_path = 'data/formula_det7/annotations/instances_test.json'
    json.dump(info_dict, open(output_path, 'w'), indent=4)
