import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import roc_auc_score

def read_excel_data(excel_path: str,root: str):
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."
    if type(excel_path) == str:
        assert os.path.exists(excel_path), f"dataset root: {excel_path} does not exist."
        excel_files = [f for f in os.listdir(excel_path) if f.endswith(('.xlsx', '.xls'))]
        excel_files.sort()
    else:
        excel_files = excel_path
        excel_files.sort()

    flower_class = [os.path.splitext(f)[0] for f in excel_files]

    class_indices = {k: v for v, k in enumerate(flower_class)}
    json_str = json.dumps({val: key for key, val in class_indices.items()}, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []
    test_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for excel_name in excel_files:
        class_name = os.path.splitext(excel_name)[0]
        image_class = class_indices[class_name]

        excel_path = os.path.join(root, excel_name)
        df = pd.read_excel(excel_path, header=None)
        folder_list = df[0].astype(str).tolist()

        images_in_this_class = []

        for sub_folder in folder_list:
            sub_folder_path = os.path.join(root, sub_folder)

            if not os.path.exists(sub_folder_path):
                print(f"Warning: Folder {sub_folder_path} defined in {excel_name} does not exist.")
                continue

            for dirpath, _, filenames in os.walk(sub_folder_path):
                for filename in filenames:
                    if os.path.splitext(filename)[-1] in supported:
                        img_path = os.path.join(dirpath, filename)
                        images_in_this_class.append(img_path)

        every_class_num.append(len(images_in_this_class))
        for img_path in images_in_this_class:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print(f"{sum(every_class_num)} images were found in the dataset.")
    print(f"Classes found: {flower_class}")

    return test_images_path, test_images_label



def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = []
        for dirpath, dirnames, filenames in os.walk(cla_path):
            for filename in filenames:
                if os.path.splitext(filename)[-1] in supported:
                    images.append(os.path.join(dirpath, filename))
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def test_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []
    test_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        images = []
        for dirpath, dirnames, filenames in os.walk(cla_path):
            for filename in filenames:
                if os.path.splitext(filename)[-1] in supported:
                    images.append(os.path.join(dirpath, filename))
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for testing.".format(len(test_images_path)))
    return test_images_path, test_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    optimizer.zero_grad()

    sample_num = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        if type(images) == list:
            sample_num += images[0].shape[0]
            images[0] = images[0].to(device)
            images[1] = images[1].to(device)
            pred = model(images)
        else:
            sample_num += images.shape[0]
            pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        labels = labels.to(device)
        true_positives += (pred_classes * labels).sum().item()
        false_positives += ((1 - labels) * pred_classes).sum().item()
        false_negatives += (labels * (1 - pred_classes)).sum().item()
        true_negatives += ((1 - labels) * (1 - pred_classes)).sum().item()

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)


        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    print("[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                              accu_loss.item() / (step + 1),
                                                              accu_num.item() / sample_num))

    precision = true_positives / (true_positives + false_positives + 0.0000001)

    recall = true_positives / (true_positives + false_negatives + 0.0000001)

    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    if (true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (
            true_negatives + false_negatives) != 0:
        mcc = (true_positives * true_negatives - false_positives * false_negatives) / (
                (true_positives + false_positives) * (true_positives + false_negatives) * (
                true_negatives + false_positives) * (true_negatives + false_negatives)) ** 0.5
    else:
        mcc = 0.0

    sensitivity = recall

    specificity = true_negatives / (true_negatives + false_positives + 0.0000001)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')
    print(f'MCC: {mcc}')
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num



def evaluate(model, data_loader, device, epoch, is_train):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    sample_num = 0

    predictions_batches = []
    labels_batches = []

    with torch.no_grad():  #
        for step, data in enumerate(data_loader):
            images, labels = data
            if isinstance(images, list):
                sample_num += images[0].shape[0]
                images = [img.to(device) for img in images]
                pred = model(images)
            else:
                sample_num += images.shape[0]
                pred = model(images.to(device))


            probs = torch.nn.functional.softmax(pred, dim=1)
            positive_probs = probs[:, 1]
            predictions_batches.append(positive_probs.cpu())
            labels_batches.append(labels.cpu())


            pred_classes = torch.max(pred, dim=1)[1]
            labels = labels.to(device)


            true_positives += (pred_classes * labels).sum().item()
            false_positives += ((1 - labels) * pred_classes).sum().item()
            false_negatives += (labels * (1 - pred_classes)).sum().item()
            true_negatives += ((1 - labels) * (1 - pred_classes)).sum().item()

            accu_num += torch.eq(pred_classes, labels).sum()
            loss = loss_function(pred, labels)
            accu_loss += loss


    all_probs = torch.cat(predictions_batches).numpy()
    all_labels = torch.cat(labels_batches).numpy()


    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)


    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch} ({is_train})')
    plt.legend(loc="lower right")


    import os
    if not os.path.exists('weights'):
        os.makedirs('weights')
    plt.savefig(f'weights/epoch_{epoch}_{is_train}_roc.png')
    plt.close()


    avg_loss = accu_loss.item() / (step + 1)
    accuracy = accu_num.item() / sample_num

    precision = true_positives / (true_positives + false_positives + 0.0000001)


    recall = true_positives / (true_positives + false_negatives + 0.0000001)


    f1_score = 2 * (precision * recall) / (precision + recall + 0.0000001)


    mcc = (true_positives * true_negatives - false_positives * false_negatives) / (
            (true_positives + false_positives) * (true_positives + false_negatives) * (
            true_negatives + false_positives) * (true_negatives + false_negatives) + 0.0000001) ** 0.5


    sensitivity = recall


    specificity = true_negatives / (true_negatives + false_positives + 0.0000001)


    print(f"[{is_train} Epoch {epoch}] Loss: {avg_loss:.3f}, Acc: {accuracy:.3f}, AUC: {roc_auc:.3f}")
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')
    print(f'MCC: {mcc}')
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')

    return avg_loss, accuracy

