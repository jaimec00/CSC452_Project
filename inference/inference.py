import tensorflow as tf
import random
import os
from pycocotools.coco import COCO
import cv2
import numpy as np


# Load the TensorFlow Keras model
#model = tf.keras.models.load_model('MODEL PATH')

# Verify the model is loaded
#model.summary()

# Get path to image directory
image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'images', 'images', 'livecell_test_images'))

# Load the COCO annotations
annotations_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'annotations'))
annFile = os.path.join(annotations_dir, "testing.json")
coco = COCO(annFile)

# Take 3 random images from the training set
n=3
imgIDs = random.sample(coco.getImgIds(), n)


for num, id in enumerate(imgIDs):
    image = coco.loadImgs(id)[0]
    image_file = image["file_name"]
    image_class = image_file.split("_")[0]
    image_path = os.path.join(image_dir, image_class, image_file)

    annIDs = coco.getAnnIds(imgIds=image['id'], catIds=[], iscrowd=None)
    anns = coco.loadAnns(annIDs)

    canvas = np.zeros((image["height"], image["width"]*3, 3), dtype=np.uint8)

    # Read the image
    img = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = img.copy()

    # Place the image in the left 1/3 of the canvas
    canvas[:image["height"], :image["width"], :] = img

    # Draw the segmentation outlines on the image from the annotations
    for ann in anns:
        segmentation = ann["segmentation"][0]
        coords = [[segmentation[i], segmentation[i+1]] for i in range(0, len(segmentation), 2)]
        coords = np.array(coords)
        coords = coords.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img_copy, [coords], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Place the image with segmentation outlines in the middle 1/3 of the canvas
    canvas[:image["height"], image["width"]:2*image["width"], :] = img_copy

    




    

    



