import os
import cv2
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt

# Adjust these paths
ANNOTATION_FILE = "C:/Users/benki/Documents/Data/LIVECell/livecell_coco_train.json"
IMAGE_DIR = "C:/Users/benki/Documents/Data/LIVECell/images/livecell_train_val_images"

def polygons_to_mask(segmentation, height, width):
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask

def build_mask(coco, image_id):
    image_info = coco.loadImgs(image_id)[0]
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    height, width = image_info['height'], image_info['width']
    final_mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        mask = polygons_to_mask(ann['segmentation'], height, width)
        final_mask = np.maximum(final_mask, mask)

    return final_mask, image_info

def display_image_and_mask(image_path, mask):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask_colored = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    combined = np.hstack((image, mask_colored))
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def menu_loop():
    coco = COCO(ANNOTATION_FILE)
    image_ids = coco.getImgIds()

    while True:
        print("\nChoose an image index (0 - {}), or 'q' to quit:".format(len(image_ids)-1))
        idx = input(">> ")

        if idx.lower() == 'q':
            break
        if not idx.isdigit() or not (0 <= int(idx) < len(image_ids)):
            print("Invalid index.")
            continue

        image_id = image_ids[int(idx)]
        mask, info = build_mask(coco, image_id)

        image_path = os.path.join(IMAGE_DIR, info['file_name'])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        display_image_and_mask(image_path, mask)

if __name__ == "__main__":
    menu_loop()
