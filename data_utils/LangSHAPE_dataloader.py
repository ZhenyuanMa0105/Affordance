# *_*coding:utf-8 *_*
"""
@File: LangSHAPE_dataloader.py
@Time: 2024/6/18 18:40$
@Author: Yaoxian
@Version: 1.0
@Contact: aarons.hdu@gmail.com
@Desc: None
"""

import os
# from mayavi import mlab
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
import glob
import random
# from scipy.spatial.transform import Rotation as R
import h5py  # 3.9.0
# import copy
from data_utils.pointcloud_utils import pc_normalize, rot_augmentation, jitter_point_cloud, shuffle_points
from tqdm import tqdm
# import open3d as o3d
from collections import defaultdict

warnings.filterwarnings('ignore')


class PartGroundingDataset(Dataset):
    def __init__(self, split='train', root='/home/yiyang/data/LangSHAPE', npoints=2048, 
                 train_mode='part-wise', class_choice=None, normal_channel=False, data_mode='full'):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.data_mode = data_mode
        self.aux_info = []
        print('GPT dataloader Partial Dataset Loader with Rotate aug: gpt4')
        self.aux_info.append('GPT dataloader Partial Dataset Loader with Rotate aug: gpt4')
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print('----1----')
        # print(self.cat)
        self.cat = {k: v for k, v in self.cat.items()}  # name--8bit_number
        self.cat2 = {v: k for k, v in self.cat.items()}  # 8bit_number--name

        # print('----2----')
        # print(self.cat2)
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        # print(self.classes_original)
        # {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8,
        # 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}
        corpus_path = self.root + '/corpus_data/' + self.data_mode + '/' + split + '_dict.json'
        with open(corpus_path, 'r') as f:
            data_str = f.read()
            self.lang_dict = json.loads(data_str)

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        if train_mode == 'part-wise':
            train_mode_tmp = train_mode + '_20240817'
        else:
            train_mode_tmp = train_mode


        self.aux_info.append(train_mode_tmp)
        if split == 'trainval':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]
        if split == 'train':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]
        if split == 'val':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]
        if split == 'test':
            with open(os.path.join(self.root, 'split_data', train_mode_tmp, split + '.txt')) as f:
                data = f.readlines()
                fns = [item.strip() for item in data]

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        # mapping from original label to share part label
        # 39-->0  39-->4
        self.shape_dict = {'0': 0, '5': 0, '11': 0, '21': 0, '35': 0, '37': 0, '39': 0, '41': 0, '1': 1, '2': 2, '3': 3,
                           '4': 4,
                           '23': 4, '33': 4, '36': 4, '6': 5, '7': 6, '8': 7, '9': 8, '12': 9, '13': 10, '31': 10,
                           '14': 11, '48': 11,
                           '15': 12, '16': 13, '17': 14, '18': 15, '19': 16, '20': 17, '22': 18, '24': 19, '26': 19,
                           '25': 20, '27': 21, '28': 22,
                           '29': 23, '30': 24, '10': 25, '32': 25, '44': 25, '34': 26, '38': 27, '40': 28, '42': 29,
                           '43': 30, '45': 31, '46': 32,
                           '47': 33, '49': 34}

        self.shape_to_label = {'body': 0, 'wing': 1, 'tail': 2, 'engine': 3, 'handle': 4, 'peak': 5, 'panel': 6,
                               'roof': 7, 'hood': 8, 'back': 9, 'seat': 10, 'leg': 11, 'arm': 12, 'earphone': 13,
                               'headband': 14, 'microphone': 15, 'head': 16, 'neck': 17, 'blade': 18, 'base': 19,
                               'shade': 20, 'tube': 21, 'keyboard': 22, 'screen': 23, 'tank': 24, 'wheel': 25,
                               'light': 26, 'barrel': 27, 'trigger': 28, 'fin': 29, 'nose': 30, 'deck': 31,
                               'bearing': 32, 'top': 33, 'drawer': 34}

        self.label_to_shape = {0: 'body', 1: 'wing', 2: 'tail', 3: 'engine', 4: 'handle', 5: 'peak', 6: 'panel',
                               7: 'roof',
                               8: 'hood', 9: 'back', 10: 'seat', 11: 'leg', 12: 'arm', 13: 'earphone', 14: 'headband',
                               15: 'microphone', 16: 'head', 17: 'neck', 18: 'blade', 19: 'base', 20: 'shade',
                               21: 'tube',
                               22: 'keyboard', 23: 'screen', 24: 'tank', 25: 'wheel', 26: 'light', 27: 'barrel',
                               28: 'trigger', 29: 'fin', 30: 'nose', 31: 'deck', 32: 'bearing', 33: 'top', 34: 'drawer'}

        # create training index
        self.datapath = []
        for fn in fns:
            cat_number = fn.split('/')[0]
            # print(self.cat2[cat_number], type(fn))
            cat = self.cat2[cat_number]
            file_name = fn.split('/')[-1]
            part = file_name.split('_')[-1]
            new_part = int(self.shape_dict[part])
            new_part_name = self.label_to_shape[new_part]
            cat_part = cat + '_' + new_part_name
            # print(cat_part)
            # print(self.lang_dict.keys())
            if cat_part.lower() in self.lang_dict.keys():
                self.datapath.append((self.cat2[cat_number], fn))


    @staticmethod
    def mapping_shape(point_seg, file_name):
        # print(point_seg)
        ground_part = int(file_name.split('_')[-1])
        for idx in range(len(point_seg)):
            # print('------1------', idx, point_seg[idx, ])
            # print(point_seg[idx], ground_part)
            if point_seg[idx] != ground_part:
                point_seg[idx] = 0
            else:
                point_seg[idx] = 1
            # point_seg[idx] = self.shape_dict[str(point_seg[idx, ])]
            # print('------2------', point_seg[idx])
        return point_seg

    def __getitem__(self, index):
        fn = self.datapath[index]
        cat = self.datapath[index][0]

        file_name = fn[1].split('/')[-1]
        part = file_name.split('_')[-1]
        new_part = int(self.shape_dict[part])
        new_part_name = self.label_to_shape[new_part]
        cat_part = cat + '_' + new_part_name

        # # get object label number
        # {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8,
        # 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}
        cls = self.classes[cat]
        cls = np.array(cls).astype(np.int32)

        # # get corpus set
        # print(self.lang_dict.keys())
        lang_data = self.lang_dict[cat_part.lower()]
        sent = random.choice(lang_data).strip()

        # # get point cloud data
        # h5_path = os.path.join(self.root, 'pd_grasp_data', fn[1], 'partial_pc_grasp.h5')
        # places = ['0', '1', '2']
        # views = ['view_1', 'view_2', 'view_3', 'view_4', 'merge']

        # with h5py.File(h5_path, 'r') as f:
        #     # pc_data = f['collect_pc'][()]
        #     place = random.choice(places)
        #     view = random.choice(views)
        #     # one view
        #     pc_data = f[place + '/' + view][()]
        # pc_data = shuffle_points(pc_data)

        # if not self.normal_channel:
        #     point_set = pc_data[:, 0:3]
        # else:
        #     point_set = pc_data[:, 0:6]

        # # # get point semantic label number
        # seg = pc_data[:, -1].astype(np.float32)
        # seg = self.mapping_shape(seg, file_name)
        # # print(type(seg), seg.shape, len(seg), np.sum(seg))
        # # print(seg)

        # point_set = rot_augmentation(point_set[:, 0:3])
        # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        # point_set = jitter_point_cloud(point_set)

        # if len(seg) < self.npoints:
        #     choice = np.random.choice(len(seg), self.npoints, replace=True)
        # else:
        #     choice = np.random.choice(len(seg), self.npoints, replace=False)

        # point_set = point_set[choice, :].transpose().astype(np.float32)
        # seg = seg[choice]
        h5_path = os.path.join(self.root, 'pd_grasp_data', fn[1], 'ch_partial_pc_grasp.h5')
        with h5py.File(h5_path, 'r') as f:
            # 获取某个数据集
            keys = list(f.keys())
            random_key = random.choice(keys)
            pc_data = f[random_key][()]
            point_set = pc_data[:, 0:3].transpose()
            seg = pc_data[:, -1]
        return point_set, cls, seg, sent, np.array(3)

    def __len__(self):
        return len(self.datapath)

# def show_grounding_part_in_object(pc_semantic, seg, color_p=(0, 0, 1), color_f = (0.7, 0.7, 0.7)):
#     mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
#     mlab.points3d(pc_semantic[:, 0], pc_semantic[:, 1], pc_semantic[:, 2], color=color_f, scale_factor=.017)
#     # color_p = (0, 0, 1) # (192/255.0,105/255.0,166/255.0)
#     pc_semantic_1 = pc_semantic[seg==1]  # 4  36
#     mlab.points3d(pc_semantic_1[:, 0], pc_semantic_1[:, 1], pc_semantic_1[:, 2], color=color_p, scale_factor=.017)
#     # mlab.show()
    
#     mlab.savefig('grouding_part.png')

def create_new_h5_files(dataset):

    for index in tqdm(range(len(dataset))):
        fn = dataset.datapath[index]
        file_name = fn[1].split('/')[-1]
        part = file_name.split('_')[-1]
        new_part = int(dataset.shape_dict[part])
        new_part_name = dataset.label_to_shape[new_part]
        cat_part = fn[0] + '_' + new_part_name

        h5_path = os.path.join(dataset.root, 'pd_grasp_data', fn[1], 'partial_pc_grasp.h5')
        places = ['0', '1', '2']
        views = ['view_1', 'view_2', 'view_3', 'view_4', 'merge']

        valid_data = []
        with h5py.File(h5_path, 'r') as f:
            for place in places:
                for view in views:
                    pc_data = f[place + '/' + view][()]
                    pc_data = shuffle_points(pc_data)

                    if not dataset.normal_channel:
                        point_set = pc_data[:, 0:3]
                    else:
                        point_set = pc_data[:, 0:6]

                    seg = pc_data[:, -1].astype(np.float32)
                    seg = dataset.mapping_shape(seg, file_name)

                    point_set = rot_augmentation(point_set[:, 0:3])
                    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
                    point_set = jitter_point_cloud(point_set)

                    if len(seg) < dataset.npoints:
                        choice = np.random.choice(len(seg), dataset.npoints, replace=True)
                    else:
                        choice = np.random.choice(len(seg), dataset.npoints, replace=False)

                    point_set = point_set[choice, :].astype(np.float32)
                    seg = seg[choice]

                    if not (np.all(seg == 0) or np.all(seg == 1)):
                        valid_data.append((point_set, seg))

        if len(valid_data) > 0:
            new_h5_path = os.path.join(dataset.root, 'pd_grasp_data', fn[1], 'ch_partial_pc_grasp.h5')
            with h5py.File(new_h5_path, 'w') as new_f:
                for i, (point_set, seg) in enumerate(valid_data):
                    combined_data = np.hstack((point_set, seg[:, np.newaxis]))
                    new_f.create_dataset(str(i), data=combined_data)

def test_grounding_dataset(vis=False):
    root = '/storage_fast/ycli/yiyang/LangSHAPE/'
    npoint = 2048
    train_mode = 'part-wise'
    normal = False
    data_mode = 'object_unknown_part_unknown'
    a = PartGroundingDataset(root=root, npoints=npoint, train_mode=train_mode, split='test',
                             normal_channel=normal, data_mode=data_mode)
    for i in range(len(a)):
        point_set, cls, seg, sent, file_name = a[i]
        print('file_name', file_name)
        print(point_set.shape)
        print('cls', cls)
        print('point class', seg, seg.shape)
        print('sent:', sent)
    # point_set, cls, seg, sent, file_name

    # print('file_name', file_name)
    # print(point_set.shape)
    # print('cls', cls)
    # print('point class', seg, seg.shape)
    # print('sent:', sent)
    # if vis:
    #     show_grounding_part_in_object(point_set,seg)

if __name__ == '__main__':
    # test_grounding_dataset(False)
    
    root = '/storage_fast/ycli/yiyang/LangSHAPE/'
    npoint = 2048
    train_mode = 'part-wise'
    normal = False
    data_mode = 'full'
    splits = ['train', 'val', 'test', 'trainval']
    for split in splits:
        print(f'Processing {split} dataset')
        a = PartGroundingDataset(root=root, npoints=npoint, train_mode=train_mode, split=split,
                                 normal_channel=normal, data_mode=data_mode)
        break