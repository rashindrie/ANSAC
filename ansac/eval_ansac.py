import sys
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torch import nn
from pathlib import Path
from os.path import join, isfile
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
from torch.utils.data import DataLoader

from datasets.WeightedIndividualDataset import WeightedIndividualDataset
from datasets.WeightedMixedDataset import WeightedMixedDataset
from core_utils.scripts import test_from_checkpoint
from core_utils.run_utils import summarize_results, seed_torch, seed_worker
from core_utils.model import ANSAC
from config import get_configs


def main(result_dir, configs, summary_csv):
    test_all_csv = f'{input_csv_dir}'
    print(f"Annotation folder: {test_all_csv}")
    print(f"Result folder: {result_dir}")

    # for tensor board visualizations
    writer = SummaryWriter(log_dir=result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_tcga = False
    if input_test_dataset == 'tcga':
        is_tcga = True

    if configs['mixed_model']:
        test_dataset = WeightedMixedDataset(img_dir=configs['feature_dir'], weights_dir=configs['weight_dir'],
                                            csv_file=test_all_csv, train=False, label_column='label',
                                            extension=configs['ext'])
    else:
        test_dataset = WeightedIndividualDataset(img_dir=configs['feature_dir'], weights_dir=configs['weight_dir'],
                                                 csv_file=test_all_csv, train=False, label_column='label',
                                                 extension=configs['ext'], is_tcga=is_tcga)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=configs['batch_size'],
                                 num_workers=configs['num_workers'], worker_init_fn=seed_worker)

    weights = None

    num_classes = 2

    model = ANSAC(num_classes=num_classes, **configs['model'])
    test_model = ANSAC(num_classes=num_classes, **configs['model'])

    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print("\nTraining {} GPUs".format(torch.cuda.device_count()))

    print("\n", configs)

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
    input_ckpt_dir = sys.argv[6]
    input_test_dataset = sys.argv[7]

    configurations = get_configs(argv=sys.argv)
    print(configurations)

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

    print(f"\n\n########################################################################################")
    print(f"########################################################################################\n\n")

    main_output_dir = f'./eval_results/{input_run_dir}/evaluation/'

    Path(main_output_dir).mkdir(parents=True, exist_ok=True)

    losses = list()
    sensitivities = list()
    specificities = list()
    fscores = list()
    weighted_accuracies = list()
    balanced_accuracies = list()
    aucs = list()

    for config_seed in configurations['seeds']:
        configurations['seed'] = config_seed

        seed_torch(device, configurations['seed'])

        output_dir = join(main_output_dir, f'seed_{config_seed}')

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        loss, sensitivity, specificity, balanced_accuracy, weighted_accuracy, fscore, auc = main(
            result_dir=output_dir, configs=configurations, summary_csv=summary_csv)

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
