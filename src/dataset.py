import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from torchvision import transforms


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CLAdapterDataset(Dataset):
    def __init__(self, *args):
        # Original public train.py calls:
        # CLAdapterDataset(is_malignant, df, val_fold, test_fold, mode, img_size, root)
        if len(args) == 8:
            _, df, val_fold, test_fold, mode, img_size, root, norm_name = args
        elif len(args) == 7:
            _, df, val_fold, test_fold, mode, img_size, root = args
            norm_name = "clip"
        elif len(args) == 3:
            mode, img_size, root = args
            df = pd.read_csv("./data/Industry/defect_supervised/yoke-suspension/anno/train.csv")
            val_fold = 0
            test_fold = 1
            norm_name = "clip"
        else:
            raise TypeError("CLAdapterDataset expects 8/7 official-train args or 3 legacy args.")

        self.mode = mode
        self.img_size = img_size
        self.root = root
        self.norm_name = norm_name
        self.df = self.select_split(df, mode, val_fold, test_fold).reset_index(drop=True)
        self.images = [path if os.path.isabs(path) else os.path.join(root, path) for path in self.df['image_path'].tolist()]
        self.labels = [int(label) for label in self.df['label'].tolist()]
        self.transforms = self.get_transforms()

    def select_split(self, df, mode, val_fold, test_fold):
        if 'split' in df.columns:
            split_name = 'val' if mode == 'valid' else mode
            return df[df['split'] == split_name].copy()
        if 'fold' in df.columns:
            if mode == 'train':
                return df[(df['fold'] != int(val_fold)) & (df['fold'] != int(test_fold))].copy()
            if mode == 'valid':
                return df[df['fold'] == int(val_fold)].copy()
            if mode == 'test':
                return df[df['fold'] == int(test_fold)].copy()
        if mode == 'train':
            return df.copy()
        return df.iloc[0:0].copy()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = self.transforms(image)
        label = torch.as_tensor(self.labels[index], dtype=torch.long)
        return image, label

    def get_transforms(self,):
        if self.norm_name == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = CLIP_MEAN, CLIP_STD
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.RandomAffine(
                    degrees=180,
                    translate=(0.05, 0.05),
                    scale=(0.70, 1.30),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transform
