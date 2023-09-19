import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import os
from hydra.utils import get_original_cwd, to_absolute_path

from inference_leaf_instance_crop import main as inference_leaf_instance_crop_main
from inference_panoptic_plant_instance import main as inference_panoptic_plant_instance_main
from inference_plant_refinement import main as inference_plant_refinement_main


@hydra.main(config_path="../configs", config_name="inference_all.yaml")
def main(inference_cfg: DictConfig):

    # inference_panoptic_plant_instance
    model_conf = OmegaConf.load(to_absolute_path(inference_cfg.panoptic_plant_segmentation_model.config))
    model_conf.model.ckpt_path = to_absolute_path(inference_cfg.panoptic_plant_segmentation_model.ckpt)
    inference_panoptic_plant_instance_main(inference_cfg, model_conf)

    # inference_plant_refinement
    model_conf = OmegaConf.load(to_absolute_path(inference_cfg.small_plant_refinement_model.config))
    model_conf.model.ckpt_path = to_absolute_path(inference_cfg.small_plant_refinement_model.ckpt)
    inference_plant_refinement_main(inference_cfg, model_conf)

    # inference_leaf_instance_crop
    model_conf = OmegaConf.load(to_absolute_path(inference_cfg.leaf_instance_crop_segmentation_model.config))
    model_conf.model.ckpt_path = to_absolute_path(inference_cfg.leaf_instance_crop_segmentation_model.ckpt)
    inference_leaf_instance_crop_main(inference_cfg, model_conf)


if __name__ == "__main__":
    main()
