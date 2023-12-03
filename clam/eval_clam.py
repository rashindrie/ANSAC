from __future__ import print_function

import os
import argparse
from datasets.IndividualDataSetClam import IndividualDataSetClam
from datasets.MixedDataSetClam import MixedCustomDataSetClam
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--weights_dir', type=str, default='', help='weights directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. ' +
                         'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 'moco'], default='small',
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False,
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False,
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, default=None)
parser.add_argument('--csv_dir', type=str, default=None, help='path to csv data folder')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)

args.n_classes = 2
print(f"Annotation folder: {args.csv_dir}")

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold + 1)

seeds = [42, 333, 2468, 1369, 2021, 21, 121, 8642, 7654, 2010]

ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(seed)) for seed in seeds]
fold_array = [seed for seed in seeds]

datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":

    args.train_mixed = False
    if args.data_root_dir == 'None_IMG':
        args.data_root_dir = {
            'cleopatra': '/data/gpfs/projects/punim1193/pmac_datasets/Cleopatra/data/featurized_wsi_imagenet/',
            'finher': '/data/gpfs/projects/punim1193/pmac_datasets/FinHer/data/featurized_wsi_imagenet/',
            'marianne': '/data/gpfs/projects/punim1193/pmac_datasets/MARIANNE/featurized_wsi_imagenet/',
            'tcga': '/data/projects/punim0512/tils_brca/TCGA/featurized_wsi_imagenet/'
        }
        args.train_mixed = True

    if args.data_root_dir == 'None_MOCO':
        args.data_root_dir = {
            'cleopatra': '/data/gpfs/projects/punim1193/pmac_datasets/Cleopatra/data/pre_processed_featurized_wsi/',
            'finher': '/data/gpfs/projects/punim1193/pmac_datasets/FinHer/data/featurized_wsi_moco92/',
            'marianne': '/data/gpfs/projects/punim1193/pmac_datasets/MARIANNE/featurized_wsi_moco92/',
            'tcga': '/data/projects/punim0512/tils_brca/TCGA/featurized_wsi_moco92/'
        }
        args.train_mixed = True

    print("\nArguments:", args)

    all_results = []
    all_auc = []
    all_acc = []
    all_sens = []
    all_spec = []
    all_f = []
    all_weight_acc = []
    all_bal_acc = []

    for ckpt_idx in range(len(ckpt_paths)):
        print(f"Checkpoint: {ckpt_paths[ckpt_idx]}")

        test_csv = f'{args.csv_dir}'

        change_ext = 'MARIANNE' in args.csv_dir or 'tiger' in args.csv_dir
        ext = '.png' if change_ext else '.ome.tif'

        is_tcga = 'TCGA' in args.data_root_dir

        if args.train_mixed:
            test_dataset = MixedCustomDataSetClam(img_dir=args.data_root_dir, csv_file=test_csv, train=False,
                                                  extension=ext)
        else:
            test_dataset = IndividualDataSetClam(img_dir=args.data_root_dir, csv_file=test_csv, train=False,
                                                 extension=ext, is_tcga=is_tcga)

        model, patient_results, test_error, auc, sens, spec, bal_acc, weight_acc, f1, fpr, tpr, thresholds, df = eval(
            test_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_sens.append(sens)
        all_spec.append(spec)
        all_weight_acc.append(weight_acc)
        all_bal_acc.append(bal_acc)
        all_f.append(f1)
        df.to_csv(os.path.join(args.save_dir, 'seed_{}.csv'.format(seeds[ckpt_idx], seeds[ckpt_idx])), index=False)

        np.save(os.path.join(args.save_dir, 'fpr_seed_{}.npy'.format(seeds[ckpt_idx])), fpr)
        np.save(os.path.join(args.save_dir, 'tpr_seed_{}.npy'.format(seeds[ckpt_idx])), tpr)
        np.save(os.path.join(args.save_dir, 'thresholds_seed_{}.npy'.format(seeds[ckpt_idx])), thresholds)

    final_df = pd.DataFrame({'seed': seeds, 'test_sensitivity': all_sens, 'test_specificity': all_spec,
                             'test_balanced_accuracies': all_bal_acc, 'test_weighted_accuracies': all_weight_acc,
                             'test_f': all_f, 'test_auc': all_auc})
    save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
