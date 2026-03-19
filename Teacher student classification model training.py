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
from model import swin_base_patch4_window7_224_in22k as create_model
from utils import *
import torchvision
from torch import nn
from tqdm import tqdm
from train import get_model
from logger import *


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.fc3 = nn.Linear(input_size // 4, num_classes)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class Teacher_student_classifier(nn.Module):
    def __init__(self, teacher_model, student_model, classifier):
        super(Teacher_student_classifier, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.classifier = classifier

    def forward(self, x):
        s_f = self.student_model(x)
        t_f = self.teacher_model(x)
        features = torch.cat((t_f, t_f - s_f), dim=1)
        # features = torch.cat((t_f, s_f), dim=1)
        output = self.classifier(features)
        return output


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    train_images_path, train_images_label = read_excel_data(args.excel_path, args.data_path)
    val_images_path, val_images_label = read_excel_data(args.val_excel_path, args.data_path)
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

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)
    teacher_model, num_features = get_model(args.teacher_model_name, args.num_classes, is_extractor=True)
    student_model = get_model(args.student_model_name, args.num_classes, is_extractor=True, num_features=num_features)
    input_size = num_features * 2
    teacher_model.load_state_dict(torch.load(args.teacher_path), strict=False)
    student_model.load_state_dict(torch.load(args.student_path), strict=False)
    teacher_model.eval()
    student_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in student_model.parameters():
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    num_classes = args.num_classes
    classifier = Classifier(input_size, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model = Teacher_student_classifier(teacher_model, student_model, classifier)
    print(args.teacher_model_name)
    print(args.student_model_name)
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch, is_train='val')
        test_loss, test_acc = evaluate(model=model,
                                       data_loader=test_loader,
                                       device=device,
                                       epoch=epoch, is_train='test')

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "test_loss", "test_acc"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[5], test_loss, epoch)
        tb_writer.add_scalar(tags[6], test_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)


        print("weights/s_t_c_model-{}.pth".format(epoch))
        torch.save(model.state_dict(), "weights/s_t_c_model-{}.pth".format(epoch))

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Teacher-Student Networks')
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for the custom loss function')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--teacher_path', type=str, default="weights/teacher-model-X.pth", help='Teacher model path')
    parser.add_argument('--student_path', type=str, default="weights/student_model.pth", help='Student Model Path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_path', type=str,
                        default="/images")
    parser.add_argument('--excel_path', type=str,
                        default=["train_classA.xlsx",
                                 "train_classB.xlsx"])
    parser.add_argument('--val_excel_path', type=str,
                        default=["val_classA.xlsx",
                                 "val_classB.xlsx"])
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


