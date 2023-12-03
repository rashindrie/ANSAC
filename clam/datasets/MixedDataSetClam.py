import sys

sys.path.append('../')
from clam.datasets.IndividualDataSetClam import IndividualDataSetClam


class MixedCustomDataSetClam(IndividualDataSetClam):
    def __init__(self, img_dir, csv_file, train=False, label_column='label', extension='.ome.tif'):
        super().__init__(img_dir, csv_file, train, label_column, extension)

    def __getbarcode__(self, idx):
        barcode = self.slide_data.barcode.iloc[idx]
        dataset = self.slide_data.dataset.iloc[idx]
        featurized_dir = self.featurized_wsi_dirs[dataset]

        if dataset == 'cleopatra':
            extension = '.ome.tif'
        elif dataset == 'finher':
            extension = '.ome.tif'
        elif dataset == 'marianne':
            extension = '.png'
        elif dataset == 'tcga':
            extension = '.png'
            barcode = self.all_images[self.all_image_barcodes.index(barcode)][:-4]
        else:
            exit(1)

        full_image_name = barcode + extension
        image_dirpath = f'{featurized_dir}/{full_image_name}/'
        return barcode, image_dirpath
