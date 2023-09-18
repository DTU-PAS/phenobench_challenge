import numpy as np
import torch
from phenobench.phenobench_loader import PhenoBench
import albumentations as A
from transformers import MaskFormerImageProcessor
from matplotlib import pyplot as plt

class PhenoBenchPanopticsPlants(PhenoBench):
    def __init__(self, root_dir, split, processor, mode="panoptic", target_type="plant_instances", transform=None, overfit=False, blackout=False):
        super().__init__(root_dir, split, target_types=["semantics", target_type], make_unique_ids=False)
        self.target_type = target_type
        self.processor = processor
        self.transform = transform
        self.overfit = overfit
        self.mode = mode
        self.blackout = blackout
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sem_mask = sample["semantics"]
        plant_mask = sample[self.target_type]
        image = np.array(sample["image"])

        # Making plant instances consecutive here, because bug in PhenoBenchq
        def replace(array: np.array, values, replacements):
            temp_array = array.copy()

            for v, r in zip(values, replacements):
                temp_array[array == v] = r

            array = temp_array
            return array
        
        instance_ids = np.unique(plant_mask)[1:]
        plant_mask = replace(plant_mask, instance_ids, np.arange(1, len(instance_ids) + 1))

        # instance_ids = np.unique(plant_mask)
        if self.mode == "panoptic":
            instance_ids = np.arange(0, len(instance_ids) + 1)
        elif self.mode == "instance":
            instance_ids = np.arange(1, len(instance_ids) + 1)
        inst2class = {}
        for instance_id in instance_ids:
            category = np.argmax(np.bincount(sem_mask[plant_mask == instance_id]))
            inst2class[instance_id] = category

        # apply transforms (need to be applied on RGB values)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=plant_mask)
            image, plant_mask = transformed['image'], transformed['mask']
            if self.blackout:
                image[plant_mask == 0] = [0, 0, 0]
            # convert to C, H, W
            image = image.transpose(2, 0, 1)
    

        if np.sum(plant_mask) == 0:
            # If the image has no objects then it is skipped
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros(
                (0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1])
            )
        else:
            inputs = self.processor([image], [plant_mask], instance_id_to_semantic_id=inst2class, return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        inputs["image_name"] = sample["image_name"]

        return inputs


    def __len__(self):
        if self.overfit:
            return 1
        else:
            return super().__len__()

if __name__ == '__main__':
    DS_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    DS_STD = np.array([58.395, 57.120, 57.375]) / 255

    image_transform = A.Compose([
        A.OneOf([A.Resize(width=512, height=512),
        A.RandomResizedCrop(height=512, width=512, scale=(0.2, 1.0), p=1.0),], p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(p=0.5),
        A.RGBShift(p=1.0),
        A.RandomBrightness(p=1.0),
        A.RandomContrast(p=1.0),
        A.OneOf([A.GaussianBlur(p=1.0),
                 A.Blur(p=1.0),
                 A.MedianBlur(p=1.0),
                 A.MotionBlur(p=1.0),
                 ], p=0.4),
        A.Normalize(mean=DS_MEAN, std=DS_STD),
    ])

    dataset_root = "/PhenoBench"

    processor = MaskFormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)

    train_dataset = PhenoBenchPanopticsPlants(dataset_root, "train", processor, transform=image_transform, blackout=True, target_type="leaf_instances")
    val_dataset = PhenoBenchPanopticsPlants(dataset_root, "val", processor, transform=image_transform, blackout=True, target_type="leaf_instances")

    image_index = 0
    grid = (3, 5)
    fig, ax = plt.subplots(*grid)
    for i in range(grid[0]):
        for j in range(grid[1]):
            inputs = train_dataset[0]

            unnormalized_image = (inputs["pixel_values"].numpy() * np.array(DS_STD)[:, None, None]) + np.array(DS_MEAN)[:, None, None]
            unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
            unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)

            ax[i, j].imshow(unnormalized_image)

    plt.show()