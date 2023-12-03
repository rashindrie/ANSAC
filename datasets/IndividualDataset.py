import sys
sys.path.append('../')

import torch
import random
import numpy as np

from os.path import join
from datasets.BaseLoader import BaseLoader


class IndividualDataset(BaseLoader):
    def __init__(self, img_dir, weights_dir, csv_file, train=False, label_column='label', extension='.ome.tif',
                 is_tcga=False):
        super().__init__(img_dir, weights_dir, csv_file, train, label_column, extension, is_tcga)

    def __getbarcode__(self, idx):
        barcode = self.data.barcode.iloc[idx]

        if self.is_tcga:
            barcode = self.all_images[self.all_image_barcodes.index(barcode)][:-4]

        full_image_name = barcode + self.extension
        image_dirpath = f'{self.featurized_wsi_dir}/{full_image_name}/'
        return barcode, image_dirpath

    def __getitem__(self, idx):
        label = self.data[self.label_column].iloc[idx]
        sTils_score = self.data.Stromal_Tils.iloc[idx]
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
        

        return features, label, barcode, sTils_score


