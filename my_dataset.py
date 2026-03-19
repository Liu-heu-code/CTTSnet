from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MyDataSet_multimodal(Dataset):

    def __init__(self, images_path: list, images_class: list, excel_path: str, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.excel_data = pd.read_excel(excel_path, engine='openpyxl')
        scaler = StandardScaler()
        self.normalized_data = scaler.fit_transform(self.excel_data)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        if img.mode != 'RGB':
            img = img.convert('RGB')

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
        row_index = self.excel_data.loc[self.excel_data['Excel medical record number column name'] == int(self.images_path[item].split('\\')[-2])].index[0]
        physiological = torch.tensor(self.normalized_data[row_index][1:], dtype=torch.float32)
        return img, physiological, label

    @staticmethod
    def collate_fn(batch):

        images, physiological, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        physiological = torch.stack(physiological, dim=0)
        labels = torch.as_tensor(labels)

        return (images, physiological), labels

class MyDataSet(Dataset):


    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
