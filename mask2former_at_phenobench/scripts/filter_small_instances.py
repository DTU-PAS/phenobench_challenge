import os
import numpy as np
from PIL import Image
import glob
import random
from tqdm import tqdm#
def main():
    dataset_path = "/PhenoBench"
    split = "val"

    image_files = glob.glob(f"{dataset_path}/{split}/images/*.png")
    random.shuffle(image_files)

    plant_size_thresh = 2500

    out_list = []
    for image_file in tqdm(image_files):
        image = np.array(Image.open(image_file))
        plant_instances = np.array(Image.open(image_file.replace("images", "plant_instances")))
        semantics = np.array(Image.open(image_file.replace("images", "semantics")))

        instance_ids = np.unique(plant_instances)[1:]

        for instance_id in instance_ids:
            instance_mask = plant_instances == instance_id
            if np.sum(instance_mask) > plant_size_thresh:
                continue
            # get rect of instance
            ys, xs = np.where(instance_mask)
            y_min, y_max = np.min(ys), np.max(ys)
            x_min, x_max = np.min(xs), np.max(xs)

            if y_max >= image.shape[0]-1 or x_max >= image.shape[1]-1 or y_min == 0 or x_min == 0:
                continue

            # Blow up rect
            f = 3.0
            h = y_max - y_min
            w = x_max - x_min
            y_c = y_min + h/2
            x_c = x_min + w/2
            size = max(h, w) * f
            y_min = max(0, int(y_c - size/2))
            y_max = min(image.shape[0], int(y_c + size/2))
            x_min = max(0, int(x_c - size/2))
            x_max = min(image.shape[1], int(x_c + size/2))

            # crop image
            cropped_mask = instance_mask[y_min:y_max, x_min:x_max]

            cropped_semantics = semantics[y_min:y_max, x_min:x_max] > 0
            if np.sum(cropped_semantics) != np.sum(cropped_mask):
                continue
            
            out_list.append(f"{os.path.basename(image_file)} {instance_id} {x_min} {y_min} {x_max} {y_max} ")
    
    with open(f"{dataset_path}/{split}/small_instances_thesh_{plant_size_thresh}.txt", "w") as f:
        f.write("\n".join(out_list))


if __name__ == "__main__":
    main()



