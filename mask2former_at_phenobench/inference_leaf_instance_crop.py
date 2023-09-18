
import lightning.pytorch as pl
import albumentations as A
import torch
import os
import numpy as np
import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from utils.helper import get_transforms_leaf_instance_crops, blow_up_rect
from transformers import Mask2FormerImageProcessor
from models.mask2former_phenobench_panoptic import Mask2FormerPhenobenchPanopticLightningModule
from PIL import Image
from phenobench.evaluation.auxiliary.panoptic_eval import PanopticQuality
import glob
import random
import tqdm

def print_metrics(m):
    metrics = {}
    pq_per_class = m["PQ"].panoptic_qualities
    metrics["PQ"] = {}
    metrics["PQ"]["mean"] = round(m["PQ"].average_pq(pq_per_class), 4)
    metrics["PQ"]["crop"] = round(pq_per_class[1]["pq"], 4)
    if len(pq_per_class) > 1:
        metrics["PQ"]["weed"] = round(pq_per_class[2]["pq"], 4)
    return metrics

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
    transforms = get_transforms_leaf_instance_crops(model_cfg, False)
    stage_1_prediction_path = f"{inference_cfg.output_folder}/{split}"
    if split in ["train", "val"]:
        do_metrics = True
    else:
        do_metrics = False

    # log folder
    log_folder = f"{inference_cfg.output_folder}/{split}"
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(f"{log_folder}/leaf_instances", exist_ok=True)

    # metrics tracking
    if do_metrics:
        m = {
            "PQ": PanopticQuality()
        }
        m_filtered = {
            "PQ": PanopticQuality()
        }
    
    idxs = random.choices(range(len(images)), k=50)
    idxs = range(len(images))
    for idx in tqdm.tqdm(idxs):
        image_path = images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        if do_metrics:
            gt_semantics = np.array(Image.open(image_path.replace("images", "semantics")))
            gt_semantics[gt_semantics == 3] = 1
            gt_semantics[gt_semantics == 4] = 2
            gt_instance_map = np.array(Image.open(image_path.replace("images", model_cfg.data.target_type)))
        
        pred_plant_instance_map = np.array(Image.open(f"{stage_1_prediction_path}/plant_instances/{image_path.split('/')[-1]}"))
        pred_semantics = np.array(Image.open(f"{stage_1_prediction_path}/semantics/{image_path.split('/')[-1]}"))

        # replace background with 0
        pred_plant_instance_map[pred_plant_instance_map == 0] = np.max(pred_plant_instance_map) + 1
        pred_plant_instance_map[pred_semantics == 0] = 0

        instance_ids = np.unique(pred_plant_instance_map)[1:]

        pred_leaf_instance_map = np.zeros_like(pred_plant_instance_map)
        leaf_instance_id_counter = 1

        for instance_id in instance_ids:
            instance_mask = pred_plant_instance_map == instance_id
            instance_semantics = pred_semantics * instance_mask
            if np.sum(instance_semantics) == 0:
                continue
            label = np.argmax(np.bincount(instance_semantics[instance_semantics != 0]))
            if label == 2:
                continue
            # get rect of instance
            ys, xs = np.where(instance_mask)
            y_min, y_max = np.min(ys), np.max(ys)
            x_min, x_max = np.min(xs), np.max(xs)
            w, h = x_max - x_min, y_max - y_min

            if w != 0 and h != 0:
                x_min, x_max, y_min, y_max, w, h =  blow_up_rect(1.5, x_min, x_max, y_min, y_max, image.shape)

                cropped_image = image[y_min:y_max, x_min:x_max]
                cropped_instance_mask = (instance_mask[y_min:y_max, x_min:x_max]).astype(np.uint8)
                cropped_image = cv2.resize(cropped_image, (model_cfg.data.img_size, model_cfg.data.img_size), interpolation=cv2.INTER_NEAREST_EXACT)
                cropped_image = transforms(image=cropped_image)["image"]
                if model_cfg.data.blackout:
                    cropped_instance_mask_tmp = cv2.resize(cropped_instance_mask, (model_cfg.data.img_size, model_cfg.data.img_size), interpolation=cv2.INTER_NEAREST_EXACT)
                    cropped_instance_mask_tmp = cv2.dilate(cropped_instance_mask_tmp, np.ones((3, 3), np.uint8), iterations=1)
                    cropped_image[cropped_instance_mask_tmp == 0] = [0, 0, 0]
                cropped_image = cropped_image.transpose(2, 0, 1)
                inputs = processor([cropped_image], return_tensors="pt")
                with torch.no_grad():
                    inputs.to("cuda")
                    inputs["mask_labels"] = None
                    inputs["class_labels"] = None
                    outputs = model(inputs)
                prediction = processor.post_process_panoptic_segmentation(outputs, 
                                                            target_sizes=[(h, w) for _ in range(len(inputs["pixel_values"]))], 
                                                            label_ids_to_fuse=[0],
                                                            threshold=model_cfg.inference.threshold,
                                                            mask_threshold=model_cfg.inference.mask_threshold,
                                                            overlap_mask_area_threshold=model_cfg.inference.overlap_mask_area_threshold,)

                prediction = prediction[0]["segmentation"].cpu().numpy()
                if prediction.dtype == np.float32:
                    continue

                #remap instance_ids
                prediction[prediction == 0] = np.max(prediction) + 1
                background_instance_id = np.argmax(np.bincount(prediction[cropped_instance_mask == 0]))
                prediction[prediction == background_instance_id] = 0
                prediction_enlarged = np.zeros_like(pred_plant_instance_map)
                prediction_enlarged[y_min:y_max, x_min:x_max] = prediction
                leaf_instance_ids = np.unique(prediction)[1:]
                for leaf_instance_id in leaf_instance_ids:
                    pred_leaf_instance_map[prediction_enlarged == leaf_instance_id] = leaf_instance_id_counter
                    leaf_instance_id_counter += 1

        # Prediction Improvement
        pred_leaf_instance_map_filtered = pred_leaf_instance_map.copy()
        # Remove weed and background semantics
        instance_ids = np.unique(pred_leaf_instance_map_filtered)[1:]
        for instance_id in instance_ids:
            max_label = np.argmax(np.bincount(pred_semantics[pred_leaf_instance_map_filtered == instance_id]))
            if max_label == 0 or max_label == 2:
                pred_leaf_instance_map_filtered[pred_leaf_instance_map_filtered == instance_id] = 0
        pred_leaf_instance_map_filtered[pred_semantics == 0] = 0
        pred_leaf_instance_map_filtered[pred_semantics == 2] = 0

        # Remove non-connecting instances
        instance_ids = np.unique(pred_leaf_instance_map_filtered)[1:]
        for instance_id in instance_ids:
            instance = (pred_leaf_instance_map_filtered == instance_id).astype(np.uint8)
            (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(instance, 4, cv2.CV_32S)
            if numLabels > 2:
                pred_leaf_instance_map_filtered[instance == 1] = 0
                keep = np.argmax(stats[1:, 4]) + 1
                pred_leaf_instance_map_filtered[labels == keep] = instance_id

        # Remove instances of small area
        instance_ids = np.unique(pred_leaf_instance_map_filtered)[1:]
        for instance_id in instance_ids:
            instance = (pred_leaf_instance_map_filtered == instance_id).astype(np.uint8)
            plant_instance_id = np.bincount(pred_plant_instance_map[instance == 1]).argmax()
            plant_instance = pred_plant_instance_map == plant_instance_id
            num_leafs_per_plant = len(np.unique(pred_leaf_instance_map_filtered * plant_instance)[1:])
            plant_instance_size = np.sum(plant_instance)
            avg_leaf_size = plant_instance_size/num_leafs_per_plant
            if np.sum(instance) < avg_leaf_size * 0.1:
                pred_leaf_instance_map_filtered[instance == 1] = 0

        if do_metrics:
            pred_semantics[pred_semantics == 2] = 0
            gt_semantics[gt_semantics == 2] = 0
            m["PQ"].compute_pq(torch.from_numpy(pred_semantics), torch.from_numpy(gt_semantics), torch.from_numpy(pred_leaf_instance_map), torch.from_numpy(gt_instance_map))
            m_filtered["PQ"].compute_pq(torch.from_numpy(pred_semantics), torch.from_numpy(gt_semantics), torch.from_numpy(pred_leaf_instance_map_filtered), torch.from_numpy(gt_instance_map))

        cv2.imwrite(f"{log_folder}/leaf_instances/{image_path.split('/')[-1]}", pred_leaf_instance_map_filtered.astype(np.uint64))

    if do_metrics:
        print(print_metrics(m))
        print(print_metrics(m_filtered))

@hydra.main(config_path="../configs", config_name="inference_all.yaml")
def run_main(inference_cfg: DictConfig):
    model_conf = OmegaConf.load(to_absolute_path(inference_cfg.leaf_instance_crop_segmentation.config))
    model_conf.model.ckpt_path = to_absolute_path(inference_cfg.leaf_instance_crop_segmentation.ckpt)
    main(inference_cfg, model_conf)

if __name__ == "__main__":
    run_main()