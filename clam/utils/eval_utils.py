import pandas as pd

from clam.models.model_mil import MIL_fc, MIL_fc_mc
from clam.models.model_clam import CLAM_SB, CLAM_MB
from clam.utils.utils import *
from clam.utils.core_utils import Accuracy_Logger

from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


def calc_metrics(true_batch_labels, pred_batch_labels, pred_probabilities, classes=[0, 1]):
    n_classes = len(classes)
    
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(true_batch_labels, pred_batch_labels).ravel()

        auc_score = roc_auc_score(true_batch_labels, pred_probabilities[:, 1])
        fpr, tpr, thresholds = roc_curve(true_batch_labels, pred_probabilities[:, 1])

        sensitivity = tp / (fn + tp)
        specificity = tn / (tn + fp)
        balanced_accuracy = (0.5 * sensitivity) + (0.5 * specificity)
        weighted_accuracy = (0.6 * sensitivity) + (0.4 * specificity)

        print("\tTN, FP, FN, TP:", tn, fp, fn, tp)

    else:
        conf_matrix_results = multilabel_confusion_matrix(true_batch_labels, pred_batch_labels, labels=classes)

        sensitivity = 0
        specificity = 0
        balanced_accuracy = 0
        weighted_accuracy = 0

        for i in range(len(conf_matrix_results)):
            tn_class_level, fp_class_level, fn_class_level, tp_class_level = conf_matrix_results[i].ravel()

            sensitivity_class_level = tp_class_level / (fn_class_level + tp_class_level)
            specificity_class_level = tn_class_level / (tn_class_level + fp_class_level)
            balanced_accuracy_class_level = (0.5 * sensitivity_class_level) + (0.5 * specificity_class_level)
            weighted_accuracy_class_level = (0.6 * sensitivity_class_level) + (0.4 * specificity_class_level)

            sensitivity += sensitivity_class_level
            specificity += specificity_class_level
            balanced_accuracy += balanced_accuracy_class_level
            weighted_accuracy += weighted_accuracy_class_level

            print(
                f"\tClass {i}: sensitivity: {sensitivity_class_level:.4f} specificity: {specificity_class_level:.4f} balanced_accuracy: {balanced_accuracy_class_level:.4f} weighted_accuracy: {weighted_accuracy_class_level:.4f}")

        sensitivity /= n_classes
        specificity /= n_classes
        balanced_accuracy /= n_classes
        weighted_accuracy /= n_classes

        aucs = []
        binary_labels = label_binarize(true_batch_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in true_batch_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], pred_probabilities[:, class_idx])
                aucs.append(auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        if micro_average:
            binary_labels = label_binarize(true_batch_labels, classes=[i for i in range(n_classes)])
            fpr, tpr, _ = roc_curve(binary_labels.ravel(), pred_probabilities.ravel())
            auc_score = auc(fpr, tpr)
        else:
            auc_score = np.nanmean(np.array(aucs))
        
    f1 = f1_score(true_batch_labels, pred_batch_labels, average='weighted')

    return sensitivity, specificity, f1, balanced_accuracy, weighted_accuracy, auc_score, fpr, tpr, thresholds
 


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, sens, spec, balanced_acc, weighted_acc, f1, fpr, tpr, thresholds, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, sens, spec, balanced_acc, weighted_acc, f1, fpr, tpr, thresholds, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    # losses = []
    actuals = []
    predictions = []

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['barcode']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'barcode': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    sensitivity, specificity, f1, balanced_accuracy, weighted_accuracy, auc_score, fpr, tpr, thresholds = calc_metrics(
        all_labels, all_preds, all_probs, classes=list(range(args.n_classes)))
    print(f"Test: \tsensitivity: {sensitivity} \tspecificity: {specificity} \tbalanced_acc: {balanced_accuracy} \tweighted_acc: {weighted_accuracy}")

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, sensitivity, specificity, balanced_accuracy, weighted_accuracy, f1, fpr, tpr, thresholds, df, acc_logger
    
