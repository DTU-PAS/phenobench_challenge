import os
import numpy as np
from PIL import Image
import glob
import random
from tqdm import tqdm
def main():
    dataset_path = "/PhenoBench"
    split = "train"

    image_files = glob.glob(f"{dataset_path}/{split}/images/*.png")
    random.shuffle(image_files)

    out_list = []
    for image_file in tqdm(image_files):
        image = np.array(Image.open(image_file))
        plant_instances = np.array(Image.open(image_file.replace("images", "plant_instances")))
        semantics = np.array(Image.open(image_file.replace("images", "semantics")))

        instance_ids = np.unique(plant_instances)

        for instance_id in instance_ids:
            instance_mask = plant_instances == instance_id
            category = np.argmax(np.bincount(semantics[instance_mask]))
            if category != 1:
                continue

            # get rect of instance
            ys, xs = np.where(instance_mask)
            y_min, y_max = np.min(ys), np.max(ys)
            x_min, x_max = np.min(xs), np.max(xs)

            # Blow up rect
            f = 1.5
            h = y_max - y_min
            w = x_max - x_min
            y_c = y_min + h/2
            x_c = x_min + w/2
            size = max(h, w) * f
            y_min = max(0, int(y_c - size/2))
            y_max = min(image.shape[0], int(y_c + size/2))
            x_min = max(0, int(x_c - size/2))
            x_max = min(image.shape[1], int(x_c + size/2))

            out_list.append(f"{os.path.basename(image_file)} {instance_id} {x_min} {y_min} {x_max} {y_max} ")
    
    with open(f"{dataset_path}/{split}/crop_instances.txt", "w") as f:
        f.write("\n".join(out_list))


if __name__ == "__main__":
    main()



