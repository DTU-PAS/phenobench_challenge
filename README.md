# Our results for Challenge CVPPA@ICCV'23: Hierarchical Panoptic Segmentation of Crops and Weeds

# Setup repository
Install the relevant pip packages
```
pip install -r requirements.txt
```

# Inference
Please download the weights for each mask2former model.

1. [Panoptic plant segmention]()
2. [Small Plant Refinement]()
3. [Leaf Instance Crop Segmentation]()

Please update the pathes in configs/inference_all.yaml (dataset_root, pathes for model checkpoints, output_folder)-

Run the following to generate the predictions.
```
python mask2former_at_phenobench/inference_all.py
```


