import sys
sys.path.append('../')
import torch
import random
import numpy as np

from os.path import join
from datasets.BaseLoader import BaseLoader


class WeightedIndividualDataset(BaseLoader):
    def __init__(self, img_dir, weights_dir, csv_file, train=False, label_column='label', extension='.png',
                 is_tcga=False):
        super().__init__(img_dir, weights_dir, csv_file, train, label_column, extension, is_tcga)

    def __getbarcode__(self, idx):
        barcode = self.data.barcode.iloc[idx]

        if self.is_tcga:
            barcode = self.all_images[self.all_image_barcodes.index(barcode)][:-4]

        full_image_name = barcode + self.extension
        image_dirpath = f'{self.featurized_wsi_dir}/{full_image_name}/'
        weights_dirpath = f'{self.weights_dir}/{full_image_name}/'
        return barcode, image_dirpath, weights_dirpath

    def __getitem__(self, idx):
        label = self.data[self.label_column].iloc[idx]
        sTils_score = self.data.Stromal_Tils.iloc[idx]
        barcode, image_dirpath, weights_dirpath = self.__getbarcode__(idx)

        if self.train:
            rot_deg = random.choice(self.rot_deg_options)
            flip = random.choice(self.flip_options)
        else:
            rot_deg = 0
            flip = 'none'

        feature_filename = f'{rot_deg}_{flip}_features.npy'
        weights_filename = f'{rot_deg}_{flip}_features.npy'

        feature_path = join(image_dirpath, feature_filename)
        weights_path = join(weights_dirpath, weights_filename)

        features = np.load(feature_path)
        weights = np.load(weights_path)

        features = torch.from_numpy(features).float()
        weights = torch.from_numpy(weights).float()

        weights = weights.reshape(-1, 24, 24, 80, 80)
        weights = weights.permute(3, 4, 1, 2, 0)

        return [features, weights], label, barcode, sTils_score
