"""
given a path, reads the data and splits into train, val test, and loads the data (possibly iterable)
"""
from pathlib import Path
from pycocotools import coco
import cv2
import numpy as np 
from pycocotools import mask as maskUtils
import random
from tqdm import tqdm

class DataLoader():

    def __init__(self, train_path="data/images/images/livecell_train_val_images", test_path="data/images/images/livecell_test_images", annotations="annotations", num_train=-1, num_val=-1, num_test=-1): # -1 means all available
        self.train = Data(train_path, f"{annotations}/training.json", num_train) # commented out to deal w/ smallest set for debugging
        self.val = Data(train_path, f"{annotations}/validation.json", num_val)
        self.test = Data(test_path, f"{annotations}/testing.json", num_test)

class Data():
    def __init__(self, data_path, annotations_path, num_samples):
        self.data_path = data_path
        self.annotations = coco.COCO(annotations_path)
        self.load_data(num_samples)

    def load_data(self, num_samples):
        # everything is 520 x 704
        imgs, labels = [], [] # convert to tensor once loaded, 

        img_ids = self.annotations.getImgIds()[:num_samples] # only the first num_samples

        pbar = tqdm(total=len(img_ids), unit='imgs', unit_scale=True, desc=f"loading data")

        for img_id in img_ids:

            # image info
            img_info = self.annotations.loadImgs([img_id])[0]

            # path to img
            img_file = img_info["file_name"]
            img_dir = img_file.split("_")[0]
            img_path = f"{self.data_path}/{img_dir}/{img_file}"
            img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

            # get segmentations
            annIds = self.annotations.getAnnIds(imgIds=img_info['id'], catIds=[], iscrowd=None)
            anns = self.annotations.loadAnns(annIds)

            # anns is a list of dicts, each corresponding to single cell
            # loop through cells and update the mask
            ann_mask = np.zeros_like(img, dtype=np.bool)

            # is crowd is 0, so have list of lists, each inner list correspond to a seperate cell
            # inner list is a polygon in format [x1, y1, x2, y2, ... xn, yn]
            # plan is to creat a mask, with one meaning that the pixels are within the area enclosed by polygon, 0 is outiside
            # pycoco tools allows you to do this automatically
            for cell in anns:
                segmentations = cell["segmentation"]
                if isinstance(segmentations, dict):
                    rle = segmentations
                else:
                    rle = maskUtils.frPyObjects(segmentations, img.shape[0], img.shape[1])

                # update mask
                ann_mask |= maskUtils.decode(rle).astype(np.bool).squeeze(2) # H x W x 1 -> H x W

            imgs.append(img)
            labels.append(ann_mask)

            pbar.update(1)

        self.imgs = np.stack(imgs, axis=0)[:, :, :, None] / 255 # [H x W] -> B x H x W x 1 (normalized to 0,1)
        self.labels = np.stack(labels, axis=0)[:, :, :, None] # [H x W] -> B x H x W x1