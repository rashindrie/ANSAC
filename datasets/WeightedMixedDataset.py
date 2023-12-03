import sys
sys.path.append('../')

from datasets.WeightedIndividualDataset import WeightedIndividualDataset


class WeightedMixedDataset(WeightedIndividualDataset):
    def __init__(self, img_dir, weights_dir, csv_file, train=False, label_column='label', extension='.png'):
        super().__init__(img_dir, weights_dir, csv_file, train, label_column, extension)

    def __getbarcode__(self, idx):
        barcode = self.data.barcode.iloc[idx]
        dataset = self.data.dataset.iloc[idx]

        if dataset == 'cleopatra' or dataset == 'finher':
            feature_dir = self.featurized_wsi_dir[dataset]
            weight_dir = self.weights_dir[dataset]
            extension = '.ome.tif'

        elif dataset == 'marianne':
            feature_dir = self.featurized_wsi_dir[dataset]
            weight_dir = self.weights_dir[dataset]
            extension = '.png'

        elif dataset == 'tcga':
            feature_dir = self.featurized_wsi_dir[dataset]
            weight_dir = self.weights_dir[dataset]
            barcode = self.all_images[self.all_image_barcodes.index(barcode)][:-4]
            extension = '.png'
        else:
            feature_dir = self.featurized_wsi_dir[0]
            weight_dir = self.weights_dir[0]
            extension = self.extension

        full_image_name = barcode + extension
        image_dirpath = f'{feature_dir}/{full_image_name}/'
        weights_dirpath = f'{weight_dir}/{full_image_name}/'

        return barcode, image_dirpath, weights_dirpath
