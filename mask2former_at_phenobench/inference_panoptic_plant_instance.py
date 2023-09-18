
import lightning.pytorch as pl
import torch
import os
import numpy as np
import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from utils.helper import get_transforms
from transformers import Mask2FormerImageProcessor
from models.mask2former_phenobench_panoptic import Mask2FormerPhenobenchPanopticLightningModule
from PIL import Image
from phenobench.evaluation.auxiliary.panoptic_eval import PanopticQuality
import glob
import random
import tqdm
from torchmetrics.classification import MulticlassJaccardIndex  # type: ignore


def semantics_as_rgb(mask):
    rgb_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_img[mask == 1] = [0, 0, 255]
    rgb_img[mask == 2] = [255, 0, 0]
    return rgb_img

def instances_as_rgb(mask):
    rgb_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    instance_ids = np.unique(mask)[1:]
    for instance_id in instance_ids:
        rgb_img[mask == instance_id] = np.random.randint(0, 255, size=3)
    return rgb_img

def main(inference_cfg: DictConfig, model_cfg : DictConfig):
    pl.seed_everything(model_cfg.seed)

    # model
    processor = Mask2FormerImageProcessor(size=(model_cfg.data.img_size, model_cfg.data.img_size), ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    model = Mask2FormerPhenobenchPanopticLightningModule.load_from_checkpoint(model_cfg.model.ckpt_path, config=model_cfg, processor=processor)
    model.eval()

    # data
    split = inference_cfg.split
    images = sorted(glob.glob(f"{inference_cfg.dataset_root}/{split}/images/*.png"))
    transforms = get_transforms(model_cfg, False)
    if split in ["train", "val"]:
        do_metrics = True
    else:
        do_metrics = False

    # log folder
    log_folder = f"{inference_cfg.output_folder}/{split}"
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(f"{log_folder}/plant_instances", exist_ok=True)
    os.makedirs(f"{log_folder}/semantics", exist_ok=True)

    # metrics tracking
    if do_metrics:
        pq = PanopticQuality()
        iou = MulticlassJaccardIndex(num_classes=3, average=None)
    
    idxs = random.choices(range(len(images)), k=10)
    idxs = range(len(images))
    for idx in tqdm.tqdm(idxs):
        image_path = images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        if do_metrics:
            gt_semantics = np.array(Image.open(image_path.replace("images", "semantics")))
            gt_semantics[gt_semantics == 3] = 1
            gt_semantics[gt_semantics == 4] = 2
            gt_instance_map = np.array(Image.open(image_path.replace("images", model_cfg.data.target_type)))

        image = transforms(image=image)["image"]
        image = image.transpose(2, 0, 1)
        inputs = processor([image], return_tensors="pt")

        with torch.no_grad():
            inputs.to("cuda")
            inputs["mask_labels"] = None
            inputs["class_labels"] = None
            outputs = model(inputs)
        

        predictions = processor.post_process_panoptic_segmentation(outputs, 
                                                                    target_sizes=[(1024, 1024) for _ in range(len(inputs["pixel_values"]))], 
                                                                    label_ids_to_fuse=[0],
                                                                    threshold=model_cfg.inference.threshold,
                                                                    mask_threshold=model_cfg.inference.mask_threshold,
                                                                    overlap_mask_area_threshold=model_cfg.inference.overlap_mask_area_threshold,)

        pred_instance_map = predictions[0]["segmentation"].cpu().numpy()
        pred_semantics = torch.zeros(pred_instance_map.shape)
        instance_ids = np.unique(pred_instance_map)
        for instance_id in instance_ids:
            for class_info in predictions[0]["segments_info"]:
                if class_info["id"] == instance_id:
                    pred_semantics[pred_instance_map == instance_id] = class_info["label_id"]
                    break
        pred_semantics[pred_semantics > 2] = 0
 
        if do_metrics:
            pq.compute_pq(pred_semantics, torch.from_numpy(gt_semantics), torch.from_numpy(pred_instance_map), torch.from_numpy(gt_instance_map))
            iou.update(pred_semantics.cpu(), torch.from_numpy(gt_semantics))

        cv2.imwrite(f"{log_folder}/plant_instances/{image_path.split('/')[-1]}", pred_instance_map.astype(np.uint64))
        cv2.imwrite(f"{log_folder}/semantics/{image_path.split('/')[-1]}", pred_semantics.cpu().numpy().astype(np.uint64))

    if do_metrics:
        metrics = {}
        metrics["IoU"] = {}
        metrics["IoU"]["mean"] = round(float(iou.compute().mean()), 4)
        metrics["IoU"]["soil"] = round(float(iou.compute()[0]), 4)
        metrics["IoU"]["crop"] = round(float(iou.compute()[1]), 4)
        metrics["IoU"]["weed"] = round(float(iou.compute()[2]), 4)
        pq_per_class = pq.panoptic_qualities
        metrics["PQ"] = {}
        metrics["PQ"]["mean"] = round(pq.average_pq(pq_per_class), 4)
        metrics["PQ"]["crop"] = round(pq_per_class[1]["pq"], 4)
        metrics["PQ"]["weed"] = round(pq_per_class[2]["pq"], 4)
        print(metrics)
    

@hydra.main(config_path="../configs", config_name="inference_all.yaml")
def run_main(inference_cfg: DictConfig):
    model_conf = OmegaConf.load(to_absolute_path(inference_cfg.panoptic_plant_segmentation_model.config))
    model_conf.model.ckpt_path = to_absolute_path(inference_cfg.panoptic_plant_segmentation_model.ckpt)
    main(inference_cfg, model_conf)

if __name__ == "__main__":
    run_main()