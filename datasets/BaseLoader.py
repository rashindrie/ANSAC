import sys
import pandas as pd
import numpy as np

from os import listdir
from torch.utils.data import Dataset

sys.path.append('../')


class BaseLoader(Dataset):
    def __init__(self, img_dir, weights_dir, csv_file, train=False, label_column='label', extension='.ome.tif',
                 is_tcga=False):

        self.data = pd.read_csv(csv_file)
        self.featurized_wsi_dir = img_dir
        self.weights_dir = weights_dir
        self.label_column = label_column
        self.train = train
        self.extension = extension
        self.is_tcga = is_tcga

        self.all_images = [f for f in listdir(img_dir)]
        self.all_image_barcodes = [i.split('.')[0] for i in self.all_images]

        self.rot_deg_options = [0, 90, 180, 270]
        self.flip_options = ['none', 'horizontal', 'vertical', 'both']

        unique_classes = len(np.unique(self.get_labels().values))

        self.label_dict = {i: i for i in range(0, unique_classes)}
        self.num_classes = len(set(self.label_dict.values()))

        self.cls_ids_prep()

    def __len__(self):
        return len(self.data[self.label_column])

    def get_labels(self):
        return self.data[self.label_column]

    def getlabel(self, ids):
        return self.data[self.label_column][ids]

    def cls_ids_prep(self):
        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.data[self.label_column] == i)[0]

    def __getbarcode__(self, idx):
        pass

    def __getitem__(self, idx):
        pass
