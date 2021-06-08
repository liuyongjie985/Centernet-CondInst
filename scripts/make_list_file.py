
import os
import json
import shutil


# test_anno_dir = '/ssd2/shared/data/zypg_data/tiku/data_v6/testset_v2/annos'
# test_files = [file for file in os.listdir(test_anno_dir)]
# with open('data/formula_det7/annotations/test_file.list', 'w') as f:
#     for file in test_files:
#         f.write(file.split('.')[0] + '\n')
# with open('data/formula_det7/annotations/test_anno.list', 'w') as f:
#     for file in test_files:
#         info_dict = json.load(open(os.path.join(test_anno_dir, file)))
#         for idx in range(len(info_dict['shapes'])):
#             f.write(file.split('.')[0] + '_%d\n' %idx)


train_anno_dir = '/ssd2/shared/data/zypg_data/tiku/data_v6/trainset_v2/annos'
train_files = [file for file in os.listdir(train_anno_dir)]
with open('data/formula_det7/annotations/train_file.list', 'w') as f:
    for file in train_files:
        f.write(file.split('.')[0] + '\n')
with open('data/formula_det7/annotations/train_anno.list', 'w') as f:
    for file in train_files:
        info_dict = json.load(open(os.path.join(train_anno_dir, file)))
        for idx in range(len(info_dict['shapes'])):
            f.write(file.split('.')[0] + '_%d\n' %idx)
