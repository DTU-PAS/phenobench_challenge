import numpy as np
import torch
from phenobench.phenobench_loader import PhenoBench
import albumentations as A
from transformers import MaskFormerImageProcessor
from matplotlib import pyplot as plt
import cv2
import random

class PhenoBenchPlantCrops(PhenoBench):
    def __init__(self, root_dir, split, processor, crop_file, target_type="plant_instances", size=256, transform=None, overfit=False, blackout=False):
        super().__init__(root_dir, split, target_types=["semantics", "plant_instances", "leaf_instances"], make_unique_ids=False)
        self.target_type = target_type
        self.processor = processor
        self.transform = transform
        self.overfit = overfit
        self.blackout = blackout
        self.size = size
        # .txt file to list
        with open(f"{root_dir}/{split}/{crop_file}") as f:
            data_points = f.read().splitlines()
        self.data_points = [data_point.split(" ") [:-1] for data_point in data_points]
    
    def __getitem__(self, idx):
        data_point = self.data_points[idx]
        image_name = data_point[0]
        image_index = self.filenames.index(image_name)
        sample = super().__getitem__(image_index)

        bbox = [int(x) for x in data_point[2:]]
        instance_id = int(data_point[1])

        plant_instance_mask = sample["plant_instances"]
        plant_instance_mask = plant_instance_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        plant_instance_mask = (plant_instance_mask == instance_id).astype(np.uint8)
        if self.target_type == "leaf_instances":
            leaf_instance_mask = sample["leaf_instances"]
            leaf_instance_mask = leaf_instance_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            leaf_instance_mask[plant_instance_mask == 0] = 0
            instance_mask = leaf_instance_mask
        else:
            instance_mask = plant_instance_mask


        image = np.array(sample["image"])[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if self.target_type == "leaf_instances":
            sem_mask = sample["semantics"]
            sem_mask = sem_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            instance_ids = np.unique(instance_mask)
            inst2class = {}
            for instance_id in instance_ids:
                category = np.argmax(np.bincount(sem_mask[instance_mask == instance_id]))
                inst2class[instance_id] = category

        # apply transforms (need to be applied on RGB values)
        if self.transform is not None:
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_NEAREST_EXACT)
            instance_mask = cv2.resize(instance_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST_EXACT)
            plant_instance_mask = cv2.resize(plant_instance_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST_EXACT)
            transformed = self.transform(image=image, mask=instance_mask, mask2=plant_instance_mask)
            image, instance_mask, plant_instance_mask = transformed['image'], transformed['mask'], transformed['mask2']
            if self.blackout:
                plant_instance_mask_tmp = cv2.dilate(plant_instance_mask, np.ones((19, 19), np.uint8), iterations=1) 
                image[plant_instance_mask_tmp == 0] = [0, 0, 0]
            # convert to C, H, W
            image = image.transpose(2, 0, 1)
    

        if np.sum(instance_mask) == 0:
            # If the image has no objects then it is skipped
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
        else:
            if self.target_type == "leaf_instances":
                inputs = self.processor([image], [instance_mask], instance_id_to_semantic_id=inst2class, return_tensors="pt")
            else:
                inputs = self.processor([image], [instance_mask], return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        inputs["image_name"] = sample["image_name"]

        return inputs


    def __len__(self):
        if self.overfit:
            return 1
        else:
            return len(self.data_points)

if __name__ == '__main__':
    DS_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    DS_STD = np.array([58.395, 57.120, 57.375]) / 255

    image_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RGBShift(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(rotate_limit=90, shift_limit=0.2, scale_limit=0.0, p=1.0),
        A.Normalize(mean=DS_MEAN, std=DS_STD),
    ], additional_targets={"mask2": "mask"})

    dataset_root = "/PhenoBench"

    processor = MaskFormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)

    train_dataset = PhenoBenchPlantCrops(dataset_root, "train", processor, "crop_instances.txt", transform=image_transform, blackout=True, target_type="leaf_instances")
    val_dataset = PhenoBenchPlantCrops(dataset_root, "val", processor, "crop_instances.txt", transform=image_transform, blackout=True, target_type="leaf_instances")

    for k in range(10):
        image_index = random.randint(0, len(train_dataset))
        grid = (3, 5)
        fig, ax = plt.subplots(*grid)
        fig2, ax2 = plt.subplots(*grid)
        for i in range(grid[0]):
            for j in range(grid[1]):
                inputs = train_dataset[image_index]

                unnormalized_image = (inputs["pixel_values"].numpy() * np.array(DS_STD)[:, None, None]) + np.array(DS_MEAN)[:, None, None]
                unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
                unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

                ax[i, j].imshow(unnormalized_image)
                ax2[i, j].imshow(inputs["mask_labels"][0])

        plt.close(fig)