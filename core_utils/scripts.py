import torch
import numpy as np
import pandas as pd
from os.path import join

import sys
import time
sys.path.append('../')

from core_utils.run_utils import calc_metrics
from core_utils.pytorch_tools import adjust_learning_rate
from core_utils.load_checkpoint import save_checkpoint, load_best_checkpoint


def train(train_loader, model, criterion, optimizer, epoch, args, device, classes):
    model.train()

    losses = []
    actuals = []
    predictions = []
    probabilities = []

    end = time.time()
    for i, (images, targets, barcodes, sTils_scores) in enumerate(train_loader):
        optimizer.zero_grad()
        targets = targets.to(device)

        if args['nic']:
            images = images.to(device, dtype=torch.float)
            output = model(images)
        else:
            output, _ = model([*images, device])

        loss = criterion(output, targets)
        losses.append(loss.item())

        preds = output.argmax(dim=1, keepdim=True)
        preds = preds.squeeze(1)

        probs = output.detach().cpu().numpy()

        actuals.extend(targets.cpu())
        predictions.extend(preds.cpu())
        probabilities.extend(probs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probabilities = np.asarray(probabilities)
    
    sensitivity, specificity, f1, balanced_accuracy, weighted_accuracy, auc, fpr, tpr, thresholds = \
        calc_metrics(actuals, predictions, probabilities)

    print(
        f"Epoch: [{epoch}] \tLoss: {np.mean(losses)} \tsensitivity: {sensitivity} \tspecificity: {specificity} "
        f"\tbalanced_acc: {balanced_accuracy} \tweighted_acc: {weighted_accuracy}")

    return np.mean(losses), [i.item() for i in actuals], [i.item() for i in predictions]


def validate(val_loader, model, criterion, args, device, test=False, epoch=0, classes=None):
    if classes is None:
        classes = [0, 1]

    losses = []

    test_loss = 0
    correct = 0

    actuals = []
    predictions = []
    probabilities = []

    main_batch_predicts, main_batch_labels, main_batch_probs = [], [], []
    main_batch_barcodes, main_batch_sTils_scores = [], []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, targets, barcodes, stil_score) in enumerate(val_loader):
            targets = targets.to(device)

            # compute output
            if args['nic']:
                images = images.to(device, dtype=torch.float)
                output = model(images)
            else:
                output, _ = model([*images, device])

            loss = criterion(output, targets)
            losses.append(loss.item())
            test_loss += loss.item()

            preds = output.argmax(dim=1, keepdim=True)
            probs = output.detach().cpu().numpy()

            correct += preds.eq(targets.view_as(preds)).sum().item()

            preds = preds.squeeze(1)

            actuals.extend(targets.cpu())
            predictions.extend(preds.cpu())
            probabilities.extend(probs)

            if test:
                main_batch_predicts.append(preds.cpu())
                main_batch_labels.append(targets.cpu())
                main_batch_barcodes.append(np.asarray(barcodes))
                main_batch_sTils_scores.append(stil_score)

    probabilities = np.asarray(probabilities)

    sensitivity, specificity, f1, balanced_accuracy, weighted_accuracy, auc, fpr, tpr, thresholds = calc_metrics(actuals,
                                                                                                                 predictions,
                                                                                                                 probabilities,
                                                                                                                 )

    print(
        f"Epoch: [{epoch}] \tLoss: {np.mean(losses)} \tsensitivity: {sensitivity} \tspecificity: {specificity} "
        f"\tbalanced_acc: {balanced_accuracy} \tweighted_acc: {weighted_accuracy}")

    if test:
        main_batch_labels = np.concatenate(main_batch_labels, axis=0)
        main_batch_predicts = np.concatenate(main_batch_predicts, axis=0)
        main_batch_barcodes = np.concatenate(main_batch_barcodes, axis=0)
        main_batch_sTils_scores = np.concatenate(main_batch_sTils_scores, axis=0)

        output_df = pd.DataFrame(
            {
                'barcode': main_batch_barcodes,
                'sTIL_score': main_batch_sTils_scores,
                'label': main_batch_labels,
                'predicted': main_batch_predicts,
            })

        return np.mean(losses), sensitivity, specificity, balanced_accuracy, weighted_accuracy, f1, output_df, auc, fpr, tpr, thresholds

    return np.mean(losses), weighted_accuracy


def train_validate(configs, optimizer, criterion, train_loader, valid_dataloader, model, device, result_dir,
                   writer, classes):
    best_val_loss = 10000
    checkpoint_epoch = 0
    count = 0

    for epoch in range(configs['start_epoch'], configs['epochs']):

        adjust_learning_rate(optimizer, epoch, configs)

        print(f"\nEpoch {epoch} \tLR: {optimizer.param_groups[0]['lr']} \tEarly Stopping Counter: {count}")
        train_loss, _, _ = train(train_loader, model, criterion, optimizer, epoch, args=configs, device=device,
                                 classes=classes)

        # evaluate on validation set
        val_loss, weigthed_acc = validate(valid_dataloader, model, criterion, args=configs, device=device,
                                          epoch=epoch, classes=classes)

        writer.add_scalars(f'loss_info', {
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, epoch)

        is_best = val_loss < best_val_loss

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=f'{result_dir}/model_best.pth.tar')

        if is_best:
            best_val_loss = val_loss

            checkpoint_epoch = epoch
            count = 0
        else:
            count += 1

        if count >= configs['early_stop_count']:
            break

    return model, best_val_loss, checkpoint_epoch, count


def test_from_checkpoint(checkpoint_dir, test_model, final_model, test_dataloader, criterion, configs, device,
                         result_dir, summary_csv, classes=None):
    if classes is None:
        classes = [0, 1]
    print(f"\nTesting with checkpoint model from {checkpoint_dir}")

    # test on testing set
    test_model = load_best_checkpoint(test_model, f'{checkpoint_dir}/model_best.pth.tar')

    if test_model is None:
        test_model = final_model
        print("Testing with final model since no checkpoint was made")

    test_model.to(device)
    loss, sens, spec, balanced_acc, weighted_acc, f1, df_best_ckpt, auc, fpr, tpr, thresholds = validate(test_dataloader, 
        test_model, criterion,
        args=configs, device=device,
        test=True, classes=classes
        )
    df_best_ckpt.to_csv(join(result_dir, 'test_predictions_best_ckpt.csv'), index=False)
    
    np.save(join(result_dir, 'fpr.npy'), fpr)
    np.save(join(result_dir, 'tpr.npy'), tpr)
    np.save(join(result_dir, 'thresholds.npy'), thresholds)

    summ_df = pd.DataFrame(
        {
            'log_folder': [result_dir],
            'checkpoint_type': ['best'],
            'checkpoint_dir': [checkpoint_dir],
            'loss': [loss],
            'sensitivity': [sens],
            'pecificity': [spec],
            'balanced_acc': [balanced_acc],
            'weighted_acc': [weighted_acc],
            'f1': [f1],
            'auc': [auc]
        })
    summ_df.to_csv(summary_csv, mode='a', header=False)

    return loss, sens, spec, balanced_acc, weighted_acc, f1, auc, fpr, tpr, thresholds

