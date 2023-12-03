from __future__ import print_function
import sys
sys.path.append('../')
import os
import torch
import argparse

import pandas as pd
import numpy as np

from utils.core_utils import train
from datasets.IndividualDataSetClam import IndividualDataSetClam
from datasets.MixedDataSetClam import MixedCustomDataSetClam


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    print(args)

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []

    all_auc = []
    all_acc = []
    all_sens = []
    all_spec = []
    all_bal_acc = []
    all_weight_acc = []
    all_f = []

    print(f"\nAnnotation folder: {args.csv_dir}")

    for i in args.seeds:
        print(f"\n**************************** SEED: {i} ****************************")
        seed_torch(i)

        train_csv = f'{args.csv_dir}/train.csv'
        val_csv = f'{args.csv_dir}/validation.csv'
        test_csv = f'{args.csv_dir}/test.csv'

        ext = '.ome.tif'
        if 'MARIANNE' in args.csv_dir or 'final' in args.csv_dir:
            ext = '.png'

        is_tcga = False
        if 'final' in args.csv_dir:
            is_tcga = True

        if args.train_mixed:
            train_dataset = MixedCustomDataSetClam(img_dir=args.data_root_dir, csv_file=train_csv, train=True,
                                                   extension=ext)
            val_dataset = MixedCustomDataSetClam(img_dir=args.data_root_dir, csv_file=val_csv, train=False,
                                                 extension=ext)
            test_dataset = MixedCustomDataSetClam(img_dir=args.data_root_dir, csv_file=test_csv, train=False,
                                                  extension=ext)
        else:
            train_dataset = IndividualDataSetClam(img_dir=args.data_root_dir, csv_file=train_csv, train=True,
                                                  extension=ext, is_tcga=is_tcga)
            val_dataset = IndividualDataSetClam(img_dir=args.data_root_dir, csv_file=val_csv, train=False,
                                                extension=ext, is_tcga=is_tcga)
            test_dataset = IndividualDataSetClam(img_dir=args.data_root_dir, csv_file=test_csv, train=False,
                                                 extension=ext, is_tcga=is_tcga)

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args, save=False)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # write results to pkl
        # filename = os.path.join(args.results_dir, 'seed_{}_results.pkl'.format(i))
        # save_pkl(filename, results)

        ckpt_path = os.path.join(args.results_dir, 's_{}_checkpoint.pt'.format(i))

        print("\nTesting model for seed {}".format(i))

        model, patient_results, test_error, auc, sens, spec, bal_acc, weight_acc, f1, fpr, tpr, thresholds, df = eval(
            test_dataset, args, ckpt_path)
        all_auc.append(auc)
        all_acc.append(1 - test_error)
        all_sens.append(sens)
        all_spec.append(spec)
        all_bal_acc.append(bal_acc)
        all_weight_acc.append(weight_acc)
        all_f.append(f1)
        df.to_csv(os.path.join(args.results_dir, f'{i}/test_results_seed_{i}.csv'), index=False)

        np.save(os.path.join(args.results_dir, f'{i}/fpr.npy'), fpr)
        np.save(os.path.join(args.results_dir, f'{i}/tpr.npy'), tpr)
        np.save(os.path.join(args.results_dir, f'{i}/thresholds.npy'), thresholds)

    final_df = pd.DataFrame({'seed': args.seeds, 'test_sensitivity': all_sens, 'test_specificity': all_spec,
                             'test_balanced_accuracies': all_bal_acc, 'test_weighted_accuracies': all_weight_acc,
                             'test_f': all_f, 'test_auc': all_auc})
    save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str,
                    default='/data/gpfs/projects/punim1193/Cleopatra/data/pre_processed_featurized_wsi/',
                    help='data directory')
parser.add_argument('--weights_dir', type=str, default='', help='weights directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seeds', type=int, nargs='+', default=[1],
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                         + 'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                    help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'moco'], default='small',
                    help='size of model, does not affect mil')
parser.add_argument('--task', type=str, default=None)
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                    help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                    help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                    help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--csv_dir', type=str, default=None, help='path to csv data folder')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


args.seeds = [int(i) for i in args.seeds]

seed_torch(args.seeds[0])

encoding_size = 1024
settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seeds': args.seeds,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

args.n_classes = 2

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({'bag_weight': args.bag_weight,
                     'inst_loss': args.inst_loss,
                     'B': args.B})

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.train_mixed = False
if args.data_root_dir == 'None_IMG':
    args.data_root_dir = {
        'cleopatra': '/data/Cleopatra/data/featurized_wsi_imagenet/',
        'marianne': '/data/MARIANNE/data/featurized_wsi_imagenet/',
        'finher': '/data/FinHer/data/featurized_wsi_imagenet/',
        'tcga': '/data/TCGA/data/featurized_wsi_imagenet/'
    }
    args.train_mixed = True

if args.data_root_dir == 'None_MOCO':
    args.data_root_dir = {
        'cleopatra': '/data/Cleopatra/data/featurized_wsi_moco/',
        'marianne': '/data/MARIANNE/data/featurized_wsi_moco/',
        'finher': '/data/FinHer/data/featurized_wsi_moco/',
        'tcga': '/data/TCGA/data/featurized_wsi_moco/'
    }
    args.train_mixed = True

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
