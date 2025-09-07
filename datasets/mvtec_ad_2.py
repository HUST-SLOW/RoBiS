import random
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

def adjust_exposure(image, exposure_value=None):
    # (H, W, 3)
    image = np.float32(image)
    image = image * (2 ** exposure_value)
    image = np.clip(image, 0, 255)
    # noise add
    mean, std = 0, 15
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    image = Image.fromarray(image)
    return image

class RandomExposureAdjustment:
    def __init__(self, overexposure_factor=1.0, underexposure_factor=-1.0):
        self.overexposure_factor = overexposure_factor
        self.underexposure_factor = underexposure_factor

    def __call__(self, image):
        if random.random() > 0.5:
            image = adjust_exposure(image, random.uniform(-0.2, 0.2))
        return image
        
def get_data_transforms(size, isize, mean_train=None, std_train=None, bright_aug=False):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    if bright_aug:
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            RandomExposureAdjustment(overexposure_factor=0.5, underexposure_factor=-0.5),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.CenterCrop(isize),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTec2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, train_state, test_state, test_type):
        self.test_type = test_type
        self.train_state = train_state
        self.test_state  = test_state 
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0
        self.bright_tag = False
        self.resize_shape = (518, 518)

    def load_dataset(self):
        img_tot, gt_tot, label_tot, type_tot = [], [], [], []

        if self.train_state:
            train_good_dir = os.path.join(self.root, 'train', 'good')
            imgs = sorted(glob.glob(os.path.join(train_good_dir, '*')))
            img_tot.extend(imgs)
            gt_tot.extend([0] * len(imgs))
            label_tot.extend([0] * len(imgs))
            type_tot.extend(['good'] * len(imgs))

        if self.test_state:
            assert self.test_type is not None, "When test_state=True, test_type must be provided"
            test_dir = os.path.join(self.root, self.test_type)

            if self.test_type == 'test_public':
                for defect_type in os.listdir(test_dir):
                    d_img_dir = os.path.join(test_dir, defect_type)
                    imgs = sorted(
                        glob.glob(os.path.join(d_img_dir, "*.png")) +
                        glob.glob(os.path.join(d_img_dir, "*.jpg")) +
                        glob.glob(os.path.join(d_img_dir, "*.bmp"))
                    )
                    if defect_type == 'good':
                        img_tot.extend(imgs)
                        gt_tot.extend([0] * len(imgs))
                        label_tot.extend([0] * len(imgs))
                        type_tot.extend(['good'] * len(imgs))
                    else:
                        gts = sorted(glob.glob(os.path.join(test_dir, 'ground_truth', defect_type, '*.png')))
                        img_tot.extend(imgs)
                        gt_tot.extend(gts)
                        label_tot.extend([1] * len(imgs))
                        type_tot.extend([defect_type] * len(imgs))
            else:
                pattern = os.path.join(test_dir, "**", "*")
                imgs = sorted(
                    glob.glob(pattern + ".png", recursive=True) +
                    glob.glob(pattern + ".jpg", recursive=True) +
                    glob.glob(pattern + ".bmp", recursive=True)
                )
                img_tot.extend(imgs)
                gt_tot.extend([0] * len(imgs))
                label_tot.extend([0] * len(imgs))
                type_tot.extend(['good'] * len(imgs))

        return map(np.array, (img_tot, gt_tot, label_tot, type_tot))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
            if gt.shape[0] == 3:
                gt = gt[0:1, :, :]
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path










