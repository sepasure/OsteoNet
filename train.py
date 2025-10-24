import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (confusion_matrix, cohen_kappa_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef)
from sklearn.preprocessing import label_binarize
import torch.nn as nn

from model import efficientnetv2_m as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch


def evaluate(model, data_loader, device, criterion, num_classes, osteogenesis_class_index, epoch=0):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.max(probabilities, dim=1)[1]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted_classes.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_scores)

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    current_osteogenesis_score = float('nan')
    if y_score.size > 0 and y_score.shape[1] > osteogenesis_class_index:
        osteogenesis_probs_in_eval = y_score[:, osteogenesis_class_index]
        current_osteogenesis_score = np.mean(osteogenesis_probs_in_eval)
    elif y_score.size > 0:
        print(
            f"[Epoch {epoch + 1} Eval Warning] osteogenesis_class_index {osteogenesis_class_index} is out of bounds for y_score shape {y_score.shape}. Osteogenesis score is NaN.")
    else:
        print(f"[Epoch {epoch + 1} Eval Warning] y_score is empty. Osteogenesis score is NaN.")

    tp, fp, fn, tn = float('nan'), float('nan'), float('nan'), float('nan')
    fp_rate_val = float('nan')
    precision, recall, f1 = float('nan'), float('nan'), float('nan')
    roc_auc, prc_auc = float('nan'), float('nan')

    if num_classes == 2:
        if len(np.unique(y_true)) < 2:
            print(
                f"[Epoch {epoch + 1} Eval Warning] Only one class present in y_true. Some binary metrics may be NaN or 0.")
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0, labels=np.unique(y_true))
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0, labels=np.unique(y_true))
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0, labels=np.unique(y_true))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            if (fp + tn) > 0:
                fp_rate_val = fp / (fp + tn)
            else:
                fp_rate_val = 0.0
            try:
                if y_score.shape[1] == 2:
                    y_score_positive_class = y_score[:, 1]
                else:
                    y_score_positive_class = y_score
                roc_auc = roc_auc_score(y_true, y_score_positive_class)
                prc_auc = average_precision_score(y_true, y_score_positive_class)
            except ValueError:
                roc_auc, prc_auc = float('nan'), float('nan')
        else:
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

            cm_binary = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm_binary.size == 4:
                tn, fp, fn, tp = cm_binary.ravel()
            else:
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                tn = np.sum((y_true == 0) & (y_pred == 0))

            if (fp + tn) > 0:
                fp_rate_val = fp / (fp + tn)
            else:
                fp_rate_val = 0.0

            y_score_positive_class = y_score[:, 1] if y_score.ndim > 1 and y_score.shape[1] == 2 else y_score
            roc_auc = roc_auc_score(y_true, y_score_positive_class)
            prc_auc = average_precision_score(y_true, y_score_positive_class)

    else:
        avg_type = 'macro'
        precision = precision_score(y_true, y_pred, average=avg_type, zero_division=0, labels=list(range(num_classes)))
        recall = recall_score(y_true, y_pred, average=avg_type, zero_division=0,
                              labels=list(range(num_classes)))
        f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0, labels=list(range(num_classes)))

        cm_multi = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        class_tps, class_fps, class_fns, class_tns, class_fp_rates = [], [], [], [], []

        for i in range(num_classes):
            tp_i = cm_multi[i, i]
            fp_i = np.sum(cm_multi[:, i]) - tp_i
            fn_i = np.sum(cm_multi[i, :]) - tp_i
            tn_i = np.sum(cm_multi) - (tp_i + fp_i + fn_i)
            class_tps.append(tp_i)
            class_fps.append(fp_i)
            class_fp_rates.append(fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0.0)

        tp = np.sum(class_tps)
        fp = np.sum(class_fps)
        fp_rate_val = np.mean(class_fp_rates) if class_fp_rates else 0.0

        try:
            y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
            if y_true_binarized.shape[1] == num_classes:
                roc_auc = roc_auc_score(y_true_binarized, y_score, multi_class='ovr', average=avg_type)
                prc_auc = average_precision_score(y_true_binarized, y_score, average=avg_type)
            else:
                print(
                    f"[Epoch {epoch + 1} Eval Warning] y_true_binarized shape {y_true_binarized.shape} not matching num_classes {num_classes}. AUCs may be NaN.")
                roc_auc, prc_auc = float('nan'), float('nan')

        except ValueError as e:
            print(f"[Epoch {epoch + 1} Eval Warning] ROC/PRC AUC calculation error (multi-class): {e}")
            roc_auc, prc_auc = float('nan'), float('nan')

    metrics_results = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "prc_auc": prc_auc,
        "tp": float(tp) if not np.isnan(tp) else float('nan'),
        "fp": float(fp) if not np.isnan(fp) else float('nan'),
        "fn": float(fn) if num_classes == 2 and not np.isnan(fn) else float('nan'),
        "tn": float(tn) if num_classes == 2 and not np.isnan(tn) else float('nan'),
        "fp_rate": float(fp_rate_val) if not np.isnan(fp_rate_val) else float('nan'),
        "osteogenesis_score": current_osteogenesis_score
    }
    return metrics_results


def save_metrics_to_excel(epochs_list, train_losses, train_accuracies,
                          val_metrics_all_epochs,
                          output_file):
    data = {
        'Epoch': epochs_list,
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
    }
    if val_metrics_all_epochs:
        first_epoch_metrics = val_metrics_all_epochs[0]
        for key in first_epoch_metrics.keys():
            data[f'Validation {key.capitalize()}'] = [epoch_metrics[key] for epoch_metrics in val_metrics_all_epochs]

    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)


def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, train_output_base, fold_idx):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title(f'Fold {fold_idx + 1} Train Metrics')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title(f'Fold {fold_idx + 1} Validation Metrics')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    combined_output_filename = f"{train_output_base}_fold_{fold_idx + 1}_loss_acc.png"
    plt.savefig(combined_output_filename)
    plt.close()


def plot_additional_val_metrics(epochs_range, val_metrics_all_epochs, output_prefix, fold_idx, num_classes):
    if not val_metrics_all_epochs:
        return

    plt.figure(figsize=(18, 12))
    epochs = list(epochs_range)

    plot_keys_fig1 = ["precision", "recall", "f1", "kappa", "mcc"]

    plt.subplot(3, 3, 1)
    plt.plot(epochs, [m['roc_auc'] for m in val_metrics_all_epochs], label='ROC AUC')
    plt.plot(epochs, [m['prc_auc'] for m in val_metrics_all_epochs], label='PRC AUC')
    plt.title(f'Fold {fold_idx + 1} Validation AUCs')
    plt.xlabel('Epoch');
    plt.ylabel('AUC');
    plt.legend();
    plt.grid(True)

    for i, key in enumerate(plot_keys_fig1):
        plt.subplot(3, 3, i + 2)
        plt.plot(epochs, [m[key] for m in val_metrics_all_epochs], label=key.capitalize())
        plt.title(f'Fold {fold_idx + 1} Validation {key.capitalize()}')
        plt.xlabel('Epoch');
        plt.ylabel(key.capitalize());
        plt.legend();
        plt.grid(True)

    plot_keys_fig2 = ["tp", "fp", "fp_rate"]
    has_data_fig2 = any(not np.isnan(m[key]) for m in val_metrics_all_epochs for key in plot_keys_fig2 if key in m)

    if has_data_fig2:
        if num_classes == 2:
            plt.subplot(3, 3, 7)
            plt.plot(epochs, [m.get('tp', float('nan')) for m in val_metrics_all_epochs], label='TP')
            plt.title(f'Fold {fold_idx + 1} Validation True Positives')
            plt.xlabel('Epoch');
            plt.ylabel('Count');
            plt.legend();
            plt.grid(True)

            plt.subplot(3, 3, 8)
            plt.plot(epochs, [m.get('fp', float('nan')) for m in val_metrics_all_epochs], label='FP')
            plt.title(f'Fold {fold_idx + 1} Validation False Positives')
            plt.xlabel('Epoch');
            plt.ylabel('Count');
            plt.legend();
            plt.grid(True)

            plt.subplot(3, 3, 9)
            plt.plot(epochs, [m.get('fp_rate', float('nan')) for m in val_metrics_all_epochs], label='FP Rate')
            plt.title(f'Fold {fold_idx + 1} Validation FP Rate')
            plt.xlabel('Epoch');
            plt.ylabel('Rate');
            plt.legend();
            plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_fold_{fold_idx + 1}_additional_metrics.png")
    plt.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    if not os.path.exists("./weights"): os.makedirs("./weights")
    if not os.path.exists("./results"): os.makedirs("./results")

    initial_train_paths, initial_train_labels, initial_val_paths, initial_val_labels = read_split_data(args.data_path)
    all_images_path = np.array(initial_train_paths + initial_val_paths)
    all_images_label = np.array(initial_train_labels + initial_val_labels)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_folds_best_epoch_metrics = []

    img_size = {"s": [300, 384], "m": [384, 480], "l": [384, 480]}
    num_model = "m"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    criterion = nn.CrossEntropyLoss()

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_images_path, all_images_label)):
        print(f"\n--- Fold {fold_idx + 1}/{kf.get_n_splits()} ---")
        tb_writer = SummaryWriter(log_dir=f"runs/fold_{fold_idx + 1}")

        train_fold_paths = all_images_path[train_indices].tolist()
        train_fold_labels = all_images_label[train_indices].tolist()
        val_fold_paths = all_images_path[val_indices].tolist()
        val_fold_labels = all_images_label[val_indices].tolist()

        train_dataset = MyDataSet(images_path=train_fold_paths, images_class=train_fold_labels,
                                  transform=data_transform["train"])
        val_dataset = MyDataSet(images_path=val_fold_paths, images_class=val_fold_labels,
                                transform=data_transform["val"])

        batch_size = args.batch_size
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        print(f'Using {nw} dataloader workers every process for fold {fold_idx + 1}')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                   num_workers=nw, collate_fn=train_dataset.collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                 num_workers=nw, collate_fn=val_dataset.collate_fn)

        model = create_model(num_classes=args.num_classes).to(device)

        if args.weights != "":
            if os.path.exists(args.weights):
                weights_dict = torch.load(args.weights, map_location=device)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                     if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(load_weights_dict, strict=False)
                print(f"Loaded pre-trained weights for fold {fold_idx + 1}")
            else:
                print(
                    f"Pre-trained weights file not found: {args.weights}. Training from scratch for fold {fold_idx + 1}.")

        if args.freeze_layers:
            for name, para in model.named_parameters():
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print(f"Training head layer: {name}")

        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        train_losses_fold, train_accuracies_fold = [], []
        val_metrics_all_epochs_fold = []

        best_acc_fold = 0.
        best_epoch_metrics_this_fold = {}

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                    device=device, epoch=epoch)
            scheduler.step()

            val_metrics_dict = evaluate(model=model, data_loader=val_loader, device=device,
                                        criterion=criterion, num_classes=args.num_classes,
                                        osteogenesis_class_index=args.osteogenesis_class_index,
                                        epoch=epoch)

            print(f"[Fold {fold_idx + 1}/{kf.get_n_splits()}, Epoch {epoch + 1}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics_dict['loss']:.4f}, Val Acc: {val_metrics_dict['accuracy']:.4f}, "
                  f"Val F1: {val_metrics_dict['f1']:.4f}")

            train_losses_fold.append(train_loss)
            train_accuracies_fold.append(train_acc)
            val_metrics_all_epochs_fold.append(val_metrics_dict)

            tb_writer.add_scalar("train_loss", train_loss, epoch)
            tb_writer.add_scalar("train_acc", train_acc, epoch)
            for key, value in val_metrics_dict.items():
                if not np.isnan(value):
                    tb_writer.add_scalar(f"val_{key}", value, epoch)
            tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

            if val_metrics_dict["accuracy"] > best_acc_fold:
                best_acc_fold = val_metrics_dict["accuracy"]
                best_epoch_metrics_this_fold = val_metrics_dict.copy()
                torch.save(model.state_dict(), f"./weights/fold_{fold_idx + 1}_best_model.pth")

        all_folds_best_epoch_metrics.append(best_epoch_metrics_this_fold if best_epoch_metrics_this_fold
                                            else {k: float('nan') for k in val_metrics_all_epochs_fold[0].keys() if
                                                  val_metrics_all_epochs_fold})

        save_metrics_to_excel(list(range(1, args.epochs + 1)),
                              train_losses_fold, train_accuracies_fold,
                              val_metrics_all_epochs_fold,
                              f"./5folds_results/metrics_fold_{fold_idx + 1}.xlsx")

        plot_metrics(train_losses_fold, train_accuracies_fold,
                     [m['loss'] for m in val_metrics_all_epochs_fold],
                     [m['accuracy'] for m in val_metrics_all_epochs_fold],
                     train_output_base=f"./5folds_results/plot_train_val",
                     fold_idx=fold_idx)

        plot_additional_val_metrics(range(1, args.epochs + 1),
                                    val_metrics_all_epochs_fold,
                                    output_prefix=f"./5folds_results/",
                                    fold_idx=fold_idx,
                                    num_classes=args.num_classes)
        tb_writer.close()

    print("\n--- Cross-Validation Summary ---")

    metric_keys_for_summary = ["loss", "accuracy", "precision", "recall", "f1", "kappa", "mcc",
                               "roc_auc", "prc_auc", "tp", "fp", "fp_rate" , "osteogenesis_score"]
    if args.num_classes == 2:
        metric_keys_for_summary.extend(["fn", "tn"])

    summary_data = {'Fold': list(range(1, kf.get_n_splits() + 1))}
    for key in metric_keys_for_summary:
        col_name = f'Best Epoch Validation {key.replace("_", " ").capitalize()}'
        summary_data[col_name] = [fold_metrics.get(key, float('nan')) for fold_metrics in all_folds_best_epoch_metrics]

    summary_df = pd.DataFrame(summary_data)

    for col in summary_df.columns:
        if col != 'Fold':
            numeric_col_data = pd.to_numeric(summary_df[col], errors='coerce')
            summary_df.loc['mean', col] = numeric_col_data.mean()
            summary_df.loc['std', col] = numeric_col_data.std()

    summary_df.to_excel("./5folds_results/cross_validation_summary.xlsx", index=True)
    print("\nCross-validation summary saved to ./results/cross_validation_summary.xlsx")

    if 'Best Epoch Validation Accuracy' in summary_df.columns:
        mean_acc = summary_df.loc['mean', 'Best Epoch Validation Accuracy']
        std_acc = summary_df.loc['std', 'Best Epoch Validation Accuracy']
        print(f"Average Best Validation Accuracy across {kf.get_n_splits()} Folds: {mean_acc:.4f} +/- {std_acc:.4f}")
    if 'Best Epoch Validation F1' in summary_df.columns:
        mean_f1 = summary_df.loc['mean', 'Best Epoch Validation F1']
        std_f1 = summary_df.loc['std', 'Best Epoch Validation F1']
        print(
            f"Average Best Epoch Validation F1-score across {kf.get_n_splits()} Folds: {mean_f1:.4f} +/- {std_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str,
                        default="E:\\temp\\depp_learning\\deep-learning-for-image-processing\\deep-learning-for-image-processing-master\\data_set\\flower_data\\flower_photos")

    parser.add_argument('--weights', type=str, default='./pre_efficientnetv2-m.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--osteogenesis_class_index', type=int, default=1,
                        help='Index of the osteogenesis class in model output (default: 1 for binary [non-osteo, osteo])')
    opt = parser.parse_args()

    if "path/to/your/flower_photos" in opt.data_path:
        print("WARNING: Please update the --data-path argument to your actual dataset location.")

    main(opt)