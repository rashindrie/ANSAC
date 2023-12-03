import sys
import torch
import pandas as pd
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from os.path import join, isfile
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')

from datasets.IndividualDataset import IndividualDataset
from datasets.MixedDataset import MixedDataset
from core_utils.scripts import test_from_checkpoint
from core_utils.run_utils import summarize_results, seed_torch, seed_worker
from core_utils.model import CNN
from config import *


def main(result_dir, configs, summary_csv):
    print(f"Annotation folder: {input_csv_dir}")
    print(f"Result folder: {result_dir}")

    # for tensor board visualizations
    writer = SummaryWriter(log_dir=result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_tcga = False
    if input_test_dataset == 'tcga':
        is_tcga = True

    if configs['mixed_model']:
        test_dataset = MixedDataset(img_dir=configs['feature_dir'], weights_dir=None,
                                            csv_file=input_csv_dir, train=False, extension=configs['ext'])
    else:
        test_dataset = IndividualDataset(img_dir=configs['feature_dir'], weights_dir=None,
                                         csv_file=input_csv_dir, train=False, extension=configs['ext'],
                                         is_tcga=is_tcga)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=configs['batch_size'],
                                 num_workers=configs['num_workers'], worker_init_fn=seed_worker)

    weights = None

    model = CNN(num_classes=2, **configs['model'])
    test_model = CNN(num_classes=2, **configs['model'])

    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    writer.close()

    ckpt_seed = configs['seed']
    ckpt_dir = f'{input_ckpt_dir}/seed_{ckpt_seed}'

    test_loss, test_sensitivity, test_specificity, test_fscore, test_weighted_accuracy, test_f1, auc, fpr, tpr, thresholds = test_from_checkpoint(
        checkpoint_dir=ckpt_dir, test_model=test_model, final_model=model, test_dataloader=test_dataloader,
        criterion=criterion, configs=configs, device=device, result_dir=result_dir, summary_csv=summary_csv,
        classes=[0, 1])

    return test_loss, test_sensitivity, test_specificity, test_fscore, test_weighted_accuracy, test_f1, auc


if __name__ == "__main__":
    input_run_dir = sys.argv[1]
    input_csv_dir = sys.argv[2]
    input_data_dir = sys.argv[3]
    input_ext = sys.argv[4]
    input_ckpt_dir = sys.argv[5]
    input_test_dataset = sys.argv[6]

    configurations['feature_dir'] = input_data_dir
    configurations['csv_dir'] = input_csv_dir
    configurations['ext'] = input_ext
    configurations['ckpt_dir'] = input_ckpt_dir

    model_name = 'CNN'

    if input_data_dir == "None":
        configurations['feature_dir'] = {
            'cleopatra': '/data/Cleopatra/data/featurized_wsi/',
            'marianne': '/data/MARIANNE/data/featurized_wsi/',
            'finher': '/data/FinHer/data/featurized_wsi/',
            'tcga': '/data/TCGA/data/featurized_wsi/'
        }
        configurations['mixed_model'] = True

    Path(f'./eval_results/{input_run_dir}').mkdir(parents=True, exist_ok=True)

    # summary result
    summary_csv = f'./eval_results/{input_run_dir}/summary.csv'

    main_csv_headers = [
        'log_folder',
        'checkpoint_type',
        'checkpoint_epoch',
        'loss',
        'Sensitivity',
        'Specificity',
        'balanced_acc',
        'weighted_acc',
        'f1',
        'auc'
    ]

    summary_df = pd.DataFrame(columns=main_csv_headers)

    if not isfile(summary_csv):
        summary_df.to_csv(summary_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, configurations['seed'])

    main_output_dir = f'./eval_results/{input_run_dir}//'

    Path(main_output_dir).mkdir(parents=True, exist_ok=True)

    losses = list()
    sensitivities = list()
    specificities = list()
    balanced_accuracies = list()
    weighted_accuracies = list()
    fscores = list()
    aucs = list()

    for config_seed in configurations['seeds']:
        configurations['seed'] = config_seed
        print(configurations)

        seed_torch(device, configurations['seed'])

        output_dir = join(main_output_dir, f'seed_{config_seed}')

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        loss, sensitivity, specificity, balanced_accuracy, weighted_accuracy, fscore, auc = main(
                                                                         result_dir=output_dir,
                                                                         configs=configurations,
                                                                         summary_csv=summary_csv)

        losses.append(loss)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        fscores.append(fscore)
        weighted_accuracies.append(weighted_accuracy)
        balanced_accuracies.append(balanced_accuracy)
        aucs.append(auc)

    # summarize results
    summarize_results(
        all_loss=losses,
        all_sensitivity=sensitivities,
        all_specificity=specificities,
        all_fscores=fscores,
        all_balanced_accuracies=balanced_accuracies,
        all_weighted_accuracies=weighted_accuracies,
        all_aucs=aucs,
        results_path=main_output_dir
    )
