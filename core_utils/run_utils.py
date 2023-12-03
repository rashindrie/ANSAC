import sys

sys.path.append('../')

import csv
import time
import random, os, torch

import numpy as np

from time import strftime
from time import gmtime
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve


def seed_torch(device, seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _start_time():
    start_time = time.clock()
    return (start_time)


def _end_time():
    end_time = time.clock()
    return (end_time)


def get_execution_time(start_time, end_time):
    return (end_time - start_time)


def print_execution_time(input_time):
    return (strftime("%H:%M:%S", gmtime(int('{:.0f}'.format(float(str((input_time))))))))


def calc_metrics(true_batch_labels, pred_batch_labels, pred_probabilities):
    tn, fp, fn, tp = confusion_matrix(true_batch_labels, pred_batch_labels).ravel()

    auc_score = roc_auc_score(true_batch_labels, pred_probabilities[:, 1])
    fpr, tpr, thresholds = roc_curve(true_batch_labels, pred_probabilities[:, 1])

    sensitivity = tp / (fn + tp)
    specificity = tn / (tn + fp)
    balanced_accuracy = (0.5 * sensitivity) + (0.5 * specificity)
    weighted_accuracy = (0.6 * sensitivity) + (0.4 * specificity)

    print("\tTN, FP, FN, TP:", tn, fp, fn, tp)

    f1 = f1_score(true_batch_labels, pred_batch_labels, average='weighted')

    return sensitivity, specificity, f1, balanced_accuracy, weighted_accuracy, auc_score, fpr, tpr, thresholds


def process_metrics(values, name):
    print(f'\n\nProcessing {name} : {values}\n')
    mean, std = np.mean(values), np.std(values)
    print(f'Summary : {mean : .4f} (+/-{std :.5f})')


def summarize_results(all_loss, all_sensitivity, all_specificity,
                  all_fscores, all_balanced_accuracies, all_weighted_accuracies,
                  all_aucs, results_path=''):
    row_string = ["Loss", "Sensitivity", "Specificity", "Balanced acc", "Weighted acc", "F score", "AUC"]

    process_metrics(all_loss, "Loss")
    process_metrics(all_sensitivity, "Sensitivity")
    process_metrics(all_specificity, "Specificity")
    process_metrics(all_balanced_accuracies, "Balanced acc")
    process_metrics(all_weighted_accuracies, "Weighted acc")
    process_metrics(all_fscores, "F score")
    process_metrics(all_aucs, "AUC")

    with open(results_path + '/summary_all_runs.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(row_string)
        for j in range(len(all_loss)):
            csv_writer.writerow(
                [all_loss[j], all_sensitivity[j], all_specificity[j],
                 all_balanced_accuracies[j],
                 all_weighted_accuracies[j], all_fscores[j], all_aucs[j]])
