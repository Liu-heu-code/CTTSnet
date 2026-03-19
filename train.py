import os
import argparse
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet152, densenet161, vgg19, shufflenet_v2_x1_0, googlenet, mobilenet_v3_large
import timm
from my_dataset import MyDataSet
from model import swin_base_patch4_window7_224_in22k as create_model
from utils import *
import torchvision
from torch import nn
from logger import *

def get_model(model_name, num_classes, model_path=None, is_extractor=None, num_features=None):
    if model_name == 'resnet152':
        model = resnet152(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if is_extractor:
            if num_features is not None:
                model.fc = nn.Linear(model.fc.in_features, num_features)
            else:
                num_features = model.fc.in_features
                model.fc = nn.Identity()
                return model, num_features

    elif model_name == 'VGG':
        model = vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        if is_extractor:
            if num_features is not None:
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_features)
            else:
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Identity()
                return model, num_features
    elif model_name == 'densenet':
        model = densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        if is_extractor:
            if num_features is not None:
                model.classifier = nn.Linear(model.classifier.in_features, num_features)
            else:
                num_features = model.classifier.in_features
                model.classifier = nn.Identity()
                return model, num_features
    elif model_name == 'shufflenet':
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if is_extractor:
            if num_features is not None:
                model.fc = nn.Linear(model.fc.in_features, num_features)
            else:
                num_features = model.fc.in_features
                model.fc = nn.Identity()
                return model, num_features
    elif model_name == 'mobilenet':
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        if is_extractor:
            if num_features is not None:
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_features)
            else:
                num_features = model.classifier[3].in_features
                model.classifier[3] = nn.Identity()
                return model, num_features
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=False)
        print('*********vit*********')
        model.head = nn.Linear(model.head.in_features, num_classes)
        if is_extractor:
            if num_features is not None:
                model.head = nn.Linear(model.head.in_features, num_features)
            else:
                num_features = model.head.in_features
                model.head = nn.Identity()
                return model, num_features
    elif model_name == 'swin':
        model = create_model(num_classes=num_classes)
        if model_path != None:
            assert os.path.exists(model_path), "weights file: '{}' not exist.".format(model_path)
            weights_dict = torch.load(model_path)["model"]
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
        if is_extractor:
            if num_features is not None:
                model.head = nn.Linear(model.head.in_features, num_features)
            else:
                num_features = model.head.in_features
                model.head = nn.Identity()
                return model, num_features
    else:
        raise ValueError('Invalid model name')
    return model


def main(args):


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

    model = get_model(args.model_name, args.num_classes, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    for epoch in range(args.epochs):
        print(args.model_name)
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch, is_train='val')
        # test
        test_loss, test_acc = evaluate(model=model,
                                       data_loader=test_loader,
                                       device=device,
                                       epoch=epoch,is_train='test')
        torch.save(model.state_dict(), "weights/teacher-model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_path', type=str,
                        default=" /images")
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

    parser.add_argument('--weights', type=str,
                        default='./swin_base_patch4_window7_224_in22k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--model_name', type=str,
                        choices=['swin', 'resnet152', 'VGG', 'densenet', 'shufflenet', 'mobilenet', 'vit'],
                        default='swin')

    opt = parser.parse_args()
    record_log(file_path="runs/")
    print(opt)
    main(opt)
