
import torch
import albumentations as A
import numpy as np

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    image_name = [example["image_name"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels, "image_name": image_name}

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    
def get_transforms(cfg, train_split):
    DS_MEAN = np.array(cfg.data.ds_mean) / 255
    DS_STD = np.array(cfg.data.ds_std) / 255

    transforms = []

    if train_split:
        transforms.append(A.OneOf([A.Resize(width=cfg.data.img_size, height=cfg.data.img_size),
                                   A.RandomResizedCrop(height=cfg.data.img_size, width=cfg.data.img_size, scale=(0.2, 1.0), p=cfg.data.p_randomresizedcrop),], p=1.0),)
        transforms.append(A.HorizontalFlip(p=cfg.data.p_horizontalflip))
        transforms.append(A.VerticalFlip(p=cfg.data.p_verticalflip))
        transforms.append(A.Rotate(p=cfg.data.p_rotate))
        transforms.append(A.RGBShift(p=cfg.data.p_rgbshift))
        transforms.append(A.RandomBrightnessContrast(p=cfg.data.p_randombrightnesscontrast))
        transforms.append(A.OneOf([A.GaussianBlur(p=1.0),
                    A.Blur(p=1.0),
                    A.MedianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                    ], p=cfg.data.p_blur))

    else:
        transforms.append(A.Resize(width=cfg.data.img_size, height=cfg.data.img_size))
    transforms.append(A.Normalize(mean=DS_MEAN, std=DS_STD))

    return A.Compose(transforms)

def get_transforms_crops(cfg, train_split):
    DS_MEAN = np.array(cfg.data.ds_mean) / 255
    DS_STD = np.array(cfg.data.ds_std) / 255

    transforms = []

    if train_split:
        transforms.append(A.HorizontalFlip(p=cfg.data.p_horizontalflip))
        transforms.append(A.VerticalFlip(p=cfg.data.p_verticalflip))
        transforms.append(A.ShiftScaleRotate(rotate_limit=90, shift_limit=0.0, scale_limit=0.0, p=cfg.data.p_rotate)),
        transforms.append(A.RGBShift(p=cfg.data.p_rgbshift))
        transforms.append(A.RandomBrightnessContrast(p=cfg.data.p_randombrightnesscontrast))
        transforms.append(A.OneOf([A.GaussianBlur(p=1.0),
                    A.Blur(p=1.0),
                    A.MedianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                    ], p=cfg.data.p_blur))
    transforms.append(A.Normalize(mean=DS_MEAN, std=DS_STD))
    return transforms


def get_transforms_leaf_instance_crops(cfg, train_split):
    return A.Compose(get_transforms_crops(cfg, train_split), additional_targets={"mask2": "mask"})


def get_transforms_small_plants(cfg, train_split):
    return A.Compose(get_transforms_crops(cfg, train_split))

def blow_up_rect(f, x_min, x_max, y_min, y_max, image_shape):
    h = y_max - y_min
    w = x_max - x_min
    y_c = y_min + h/2
    x_c = x_min + w/2
    size = max(h, w) * f
    y_min = max(0, int(y_c - size/2))
    y_max = min(image_shape[0], int(y_c + size/2))
    x_min = max(0, int(x_c - size/2))
    x_max = min(image_shape[1], int(x_c + size/2))
    w, h = x_max - x_min, y_max - y_min
    return x_min, x_max, y_min, y_max, w, h
