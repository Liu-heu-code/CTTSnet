import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from Tsloss import CustomLoss
import os
import argparse
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet152, densenet161, vgg19, shufflenet_v2_x1_0, googlenet, mobilenet_v3_large
import timm
from my_dataset import MyDataSet
from model import swin_large_patch4_window7_224_in22k as create_model
from utils import *
import torchvision
from torch import nn
from tqdm import tqdm
from train import get_model
from logger import *

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    train_images_path, train_images_label = read_excel_data(args.excel_path, args.data_path)
    test_images_path, test_images_label = read_excel_data(args.test_excel_path, args.test_data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "test": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform["test"])
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)

    teacher_model, num_features = get_model(args.teacher_model_name, args.num_classes, is_extractor=True)
    student_model = get_model(args.student_model_name, args.num_classes, is_extractor=True, num_features=num_features)
    teacher_model.load_state_dict(torch.load(args.teacher_path), strict=False)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in student_model.parameters():
        param.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=0)
    accu_loss = 0
    lossfn = CustomLoss(margin=args.margin)
    sample_num = 0
    batch_num = 0
    best_loss = 100
    print(args.teacher_model_name)
    print(args.student_model_name)
    for epoch in range(args.epochs):
        for step, data in tqdm(enumerate(train_loader)):
            images, labels = data
            sample_num += images.shape[0]
            student_output = student_model(images.to(device))
            teacher_output = teacher_model(images.to(device))
            loss = lossfn(teacher_output, student_output, labels.to(device))
            loss.backward()
            accu_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            batch_num += 1
        if accu_loss / batch_num < best_loss:
            torch.save(student_model.state_dict(), 'weights/student_model.pth')
            best_loss = accu_loss / batch_num
        print("epoch:{} loss:{:.3f}".format(epoch, accu_loss / batch_num))
        batch_num = 0
        accu_loss = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Teacher-Student Networks')
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for the custom loss function')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--teacher_path', type=str, default="weights/teacher-model-x.pth", help='Teacher model path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_path', type=str,
                        default="/images")
    parser.add_argument('--excel_path', type=str,
                        default=["train_classA.xlsx",
                                 "train_classB.xlsx"])
    parser.add_argument('--test_data_path', type=str,
                        default="/images")
    parser.add_argument('--test_excel_path', type=str,
                        default=["test_classA.xlsx",
                                 "test_classB.xlsx"])
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--teacher_model_name', type=str,
                        choices=['swin', 'resnet152', 'VGG', 'densenet', 'shufflenet', 'mobilenet', 'vit'],
                        default='swin')
    parser.add_argument('--student_model_name', type=str,
                        choices=['swin', 'resnet152', 'VGG', 'densenet', 'shufflenet', 'mobilenet', 'vit'],
                        default='mobilenet')
    args = parser.parse_args()
    record_log(file_path="runs/")
    print(args)
    main(args)
