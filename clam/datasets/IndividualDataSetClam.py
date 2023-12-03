import sys
import torch
import random
import pandas as pd
import numpy as np
from os import listdir
from os.path import join
from torch.utils.data import Dataset

sys.path.append('../')


class IndividualDataSetClam(Dataset):
    def __init__(self, img_dir, csv_file, train=False, label_column='label', is_mlp=False, label_dict=None,
                 extension='.ome.tif', is_tcga=False):
        if label_dict is None:
            label_dict = {0: 0, 1: 1}
        self.slide_data = pd.read_csv(csv_file)
        self.featurized_wsi_dir = img_dir
        self.label_column = label_column
        self.train = train
        self.is_mlp = is_mlp
        self.extension = extension
        self.is_tcga = is_tcga

        self.rot_deg_options = [0, 90, 180, 270]
        self.flip_options = ['none', 'horizontal', 'vertical', 'both']

        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))

        if is_tcga:
            self.all_images = [f for f in listdir(img_dir)]
            self.all_image_barcodes = [i.split('.')[0] for i in self.all_images]

        self.cls_ids_prep()

    def __len__(self):
        return len(self.slide_data[self.label_column])

    def get_labels(self):
        return self.slide_data[self.label_column]

    def getlabel(self, ids):
        return self.slide_data[self.label_column][ids]

    def cls_ids_prep(self):
        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data[self.label_column] == i)[0]

    def __getbarcode__(self, idx):
        barcode = self.slide_data.barcode.iloc[idx]
        extension = self.extension

        if self.is_tcga:
            barcode = self.all_images[self.all_image_barcodes.index(barcode)][:-4]
            extension = '.png'

        full_image_name = barcode + extension
        image_dirpath = f'{self.featurized_wsi_dir}/{full_image_name}/'

        return barcode, image_dirpath

    def __getitem__(self, idx):
        label = self.slide_data[self.label_column].iloc[idx]
        # sTils_score = self.slide_data.Stromal_Tils.iloc[idx]

        barcode, image_dirpath = self.__getbarcode__(idx)

        if self.train:
            rot_deg = random.choice(self.rot_deg_options)
            flip = random.choice(self.flip_options)
        else:
            rot_deg = 0
            flip = 'none'

        feature_filename = f'{rot_deg}_{flip}_features.npy'
        feature_path = join(image_dirpath, feature_filename)
        features = np.load(feature_path)

        features = torch.from_numpy(features).float()
        features = features.reshape(-1, 6400).transpose(1, 0)
        return features, label
