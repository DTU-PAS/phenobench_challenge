
import lightning.pytorch as pl
import torch
import os
import numpy as np
import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from utils.helper import get_transforms_small_plants, blow_up_rect
from transformers import Mask2FormerImageProcessor
from models.mask2former_phenobench_plant_refinement import Mask2FormerPhenobenchPlantRefinementLightningModule
from PIL import Image
from phenobench.evaluation.auxiliary.panoptic_eval import PanopticQuality
import glob
import matplotlib.pyplot as plt
import tqdm
from torchmetrics.classification import MulticlassJaccardIndex  # type: ignore

def print_metrics(m):
    metrics = {}
    metrics["IoU"] = {}
    metrics["IoU"]["mean"] = round(float(m["IoU"].compute().mean()), 4)
    metrics["IoU"]["soil"] = round(float(m["IoU"].compute()[0]), 4)
    metrics["IoU"]["crop"] = round(float(m["IoU"].compute()[1]), 4)
    metrics["IoU"]["weed"] = round(float(m["IoU"].compute()[2]), 4)
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
    model = Mask2FormerPhenobenchPlantRefinementLightningModule.load_from_checkpoint(model_cfg.model.ckpt_path, config=model_cfg, processor=processor)
    model.eval()

    # data
    plant_size_thresh = 2500 # pixels
    split = inference_cfg.split
    images = sorted(glob.glob(f"{inference_cfg.dataset_root}/{split}/images/*.png"))
    transforms = get_transforms_small_plants(model_cfg, False)
    panoptic_prediction_path = f"{inference_cfg.output_folder}/{split}"
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
        m = {
            "IoU": MulticlassJaccardIndex(num_classes=3, average=None),
            "PQ": PanopticQuality()
        }
    
    idxs = range(len(images))
    for idx in tqdm.tqdm(idxs):
        image_path = images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        if do_metrics:
            gt_semantics = np.array(Image.open(image_path.replace("images", "semantics")))
            gt_semantics[gt_semantics == 3] = 1
            gt_semantics[gt_semantics == 4] = 2
            gt_instance_map = np.array(Image.open(image_path.replace("images", model_cfg.data.target_type)))
        
        pred_instance_map = np.array(Image.open(f"{panoptic_prediction_path}/plant_instances/{image_path.split('/')[-1]}"))
        pred_semantics = np.array(Image.open(f"{panoptic_prediction_path}/semantics/{image_path.split('/')[-1]}"))
        pred_instance_map_2 = np.zeros_like(pred_instance_map)
        pred_semantics_2 = np.zeros_like(pred_semantics)

        # replace background with 0
        pred_instance_map[pred_instance_map == 0] = np.max(pred_instance_map) + 1
        pred_instance_map[pred_semantics == 0] = 0

        instance_ids = np.unique(pred_instance_map)[1:]


        for instance_id in instance_ids:
            instance_mask = pred_instance_map == instance_id
            instance_semantics = pred_semantics * instance_mask
            if np.sum(instance_semantics) == 0:
                continue
            label = np.argmax(np.bincount(instance_semantics[instance_semantics != 0]))
            # get rect of instance
            ys, xs = np.where(instance_mask)
            y_min, y_max = np.min(ys), np.max(ys)
            x_min, x_max = np.min(xs), np.max(xs)
            w, h = x_max - x_min, y_max - y_min

            if np.sum(instance_mask) < plant_size_thresh and w != 0 and h != 0:

                if y_max >= instance_mask.shape[0]-1 or x_max >= instance_mask.shape[1]-1 or y_min == 0 or x_min == 0:
                    pred_instance_map_2[instance_mask] = instance_id
                    pred_semantics_2[instance_mask] = label
                    continue

                x_min, x_max, y_min, y_max, w, h =  blow_up_rect(3.0, x_min, x_max, y_min, y_max, image.shape)

                # If plant is not isolated, we skip for now
                cropped_mask = instance_mask[y_min:y_max, x_min:x_max]
                cropped_semantics = pred_semantics[y_min:y_max, x_min:x_max] > 0
                if np.sum(cropped_semantics) != np.sum(cropped_mask):
                    pred_instance_map_2[instance_mask] = instance_id
                    pred_semantics_2[instance_mask] = label
                    continue

                cropped_image = image[y_min:y_max, x_min:x_max]
                cropped_image = cv2.resize(cropped_image, (model_cfg.data.img_size, model_cfg.data.img_size), interpolation=cv2.INTER_NEAREST_EXACT)
                cropped_image = transforms(image=cropped_image)["image"]
                cropped_image = cropped_image.transpose(2, 0, 1)
                inputs = processor([cropped_image], return_tensors="pt")
                with torch.no_grad():
                    inputs.to("cuda")
                    inputs["mask_labels"] = None
                    inputs["class_labels"] = None
                    outputs = model(inputs)
                prediction = processor.post_process_semantic_segmentation(outputs, 
                                                            target_sizes=[(h, w) for _ in range(len(inputs["pixel_values"]))])
                prediction = prediction[0].cpu().numpy()

                prediction_enlarged = np.zeros_like(instance_mask)
                prediction_enlarged[y_min:y_max, x_min:x_max] = prediction
                instance_mask[pred_instance_map == 0] = 1
                prediction_enlarged[instance_mask == 0] = 0
                pred_instance_map_2[prediction_enlarged == 1] = instance_id
                pred_semantics_2[prediction_enlarged == 1] = label
            else: 
                pred_instance_map_2[instance_mask] = instance_id
                pred_semantics_2[instance_mask] = label



        if do_metrics:
            m["PQ"].compute_pq(torch.from_numpy(pred_semantics_2), torch.from_numpy(gt_semantics), torch.from_numpy(pred_instance_map_2), torch.from_numpy(gt_instance_map))
            m["IoU"].update(torch.from_numpy(pred_semantics_2), torch.from_numpy(gt_semantics))

        cv2.imwrite(f"{log_folder}/plant_instances/{image_path.split('/')[-1]}", pred_instance_map_2.astype(np.uint64))
        cv2.imwrite(f"{log_folder}/semantics/{image_path.split('/')[-1]}", pred_semantics_2.astype(np.uint64))

    if do_metrics:
        print(print_metrics(m))

@hydra.main(config_path="../configs", config_name="inference_all.yaml")
def run_main(inference_cfg: DictConfig):
    model_conf = OmegaConf.load(to_absolute_path(inference_cfg.small_plant_refinement_model.config))
    model_conf.model.ckpt_path = to_absolute_path(inference_cfg.small_plant_refinement_model.ckpt)
    main(inference_cfg, model_conf)

if __name__ == "__main__":
    run_main()