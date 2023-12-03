import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import optim, nn
from pathlib import Path
from os.path import join, isfile
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')

from torch.utils.data import DataLoader

from datasets.WeightedIndividualDataset import WeightedIndividualDataset
from datasets.WeightedMixedDataset import WeightedMixedDataset
from core_utils.scripts import train_validate, test_from_checkpoint
from core_utils.run_utils import summarize_results, seed_torch, seed_worker
from core_utils.model import ANSAC
from core_utils.CustomSampler import BalancedBatchSampler
from config import get_configs


def main(model_name, result_dir, configs, summary_csv):
    print(f"Annotation folder: {input_csv_dir}")
    print(f"Result folder: {result_dir}")

    # for tensor board visualizations
    writer = SummaryWriter(log_dir=result_dir)

    train_csv = f'{input_csv_dir}/train.csv'
    val_csv = f'{input_csv_dir}/validation.csv'
    test_csv = f'{input_csv_dir}/test.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_tcga = False
    if 'final' in train_csv:
        is_tcga = True

    if configs['mixed_model']:
        train_dataset = WeightedMixedDataset(img_dir=configs['feature_dir'], weights_dir=configs['weight_dir'],
                                             csv_file=train_csv, train=True, label_column='label',
                                             extension=configs['ext'])
        val_dataset = WeightedMixedDataset(img_dir=configs['feature_dir'], weights_dir=configs['weight_dir'],
                                           csv_file=val_csv, train=False, label_column='label',
                                           extension=configs['ext'])
        test_dataset = WeightedMixedDataset(img_dir=configs['feature_dir'], weights_dir=configs['weight_dir'],
                                            csv_file=test_csv, train=False, label_column='label',
                                            extension=configs['ext'])
    else:
        train_dataset = WeightedIndividualDataset(img_dir=configs['feature_dir'], weights_dir=configs['weight_dir'],
                                                  csv_file=train_csv, train=True, label_column='label',
                                                  extension=configs['ext'], is_tcga=is_tcga)
        val_dataset = WeightedIndividualDataset(img_dir=configs['feature_dir'], weights_dir=configs['weight_dir'],
                                                csv_file=val_csv, train=False, label_column='label',
                                                extension=configs['ext'], is_tcga=is_tcga)
        test_dataset = WeightedIndividualDataset(img_dir=configs['feature_dir'], weights_dir=configs['weight_dir'],
                                                 csv_file=test_csv, train=False, label_column='label',
                                                 extension=configs['ext'], is_tcga=is_tcga)

    print("\nUsing BalancedBatchSampler unweighted CE")

    sampler = BalancedBatchSampler(train_dataset, torch.from_numpy(train_dataset.get_labels().values))

    train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, batch_size=configs['batch_size'],
                              num_workers=configs['num_workers'], worker_init_fn=seed_worker)
    valid_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=configs['batch_size'],
                                  num_workers=configs['num_workers'], worker_init_fn=seed_worker)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=configs['batch_size'],
                                 num_workers=configs['num_workers'], worker_init_fn=seed_worker)

    weights = None

    unique_classes = np.unique(train_dataset.get_labels().values)
    num_classes = len(unique_classes)

    print(f'\n Training on {model_name} model using {num_classes} classes')

    model = ANSAC(num_classes=num_classes, **configs['model'])
    test_model = ANSAC(num_classes=num_classes, **configs['model'])

    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=configs['lr'],
                          momentum=configs['momentum'],
                          weight_decay=configs['weight_decay'])

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print("\nTraining {} GPUs".format(torch.cuda.device_count()))

    print("\n", configs)

    model, best_val_loss, checkpoint_epoch, count = train_validate(configs=configs,
                                                                   optimizer=optimizer,
                                                                   criterion=criterion,
                                                                   result_dir=result_dir,
                                                                   train_loader=train_loader,
                                                                   valid_dataloader=valid_dataloader,
                                                                   model=model,
                                                                   device=device,
                                                                   writer=writer,
                                                                   classes=unique_classes)

    writer.close()

    test_loss, test_sensitivity, test_specificity, test_fscore, test_weighted_accuracy, test_f1, auc, fpr, tpr, thresholds = test_from_checkpoint(
        checkpoint_dir=result_dir, test_model=test_model, final_model=model, test_dataloader=test_dataloader,
        criterion=criterion, configs=configs, device=device, result_dir=result_dir, summary_csv=summary_csv,
        classes=unique_classes)

    return test_loss, test_sensitivity, test_specificity, test_fscore, test_weighted_accuracy, test_f1, auc


if __name__ == "__main__":
    input_run_dir = sys.argv[1]
    input_csv_dir = sys.argv[2]
    model_name = 'ANSAC'

    configurations = get_configs(argv=sys.argv)
    print(configurations)

    Path(f'./logs/{input_run_dir}').mkdir(parents=True, exist_ok=True)

    # summary result
    summary_csv = f'./logs/{input_run_dir}/summary.csv'

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

    print(f"\n\n########################################################################################")
    print(f"########################################################################################\n\n")

    main_output_dir = f'./logs/{input_run_dir}/{model_name}'

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
            model_name=model_name, result_dir=output_dir, configs=configurations, summary_csv=summary_csv)

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
