import tensorflow as tf
import random
import os
from pycocotools.coco import COCO
import cv2
import numpy as np
from zipfile import ZipFile
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from display_script import build_mask

# Path to the .zip file containing the trained model
zip_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_model', 'output.zip'))
# Target directory for unzipping
extracted_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_model', 'unzipped'))
# Unzip the model if it hasn't been unzipped yet
if not os.path.exists(extracted_dir):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

# Path to the unzipped model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_model', 'unzipped', 'output', 'model.keras'))

# Load the TensorFlow Keras model
model = tf.keras.models.load_model(model_path)

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

    canvas = np.zeros((image["height"]+50, image["width"]*3, 3), dtype=np.uint8)
    canvas[:50, :, :] = [255, 255, 255] 
    cv2.putText(canvas, "Original Image", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(canvas, "Annotation Masks", (714, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(canvas, "Predicted Masks", (1418, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Read the image
    img = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    
    img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    # Place the image in the left 1/3 of the canvas
    canvas[50:image["height"]+50, :image["width"], :] = img_copy

    # Get the segmentation mask from the image annotations
    annotation_mask = build_mask(coco, id)[0]
    annotation_mask = cv2.resize(annotation_mask, (img.shape[1], img.shape[0]))

    # Overlay the segmentation mask on the original image
    img_copy[annotation_mask == 1] = [255, 0, 0] 

    # Place the image with segmentation outlines in the middle 1/3 of the canvas
    canvas[50:image["height"]+50, image["width"]:2*image["width"], :] = img_copy

    image_normalized = img / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    # Predict the segmentation mask
    predicted_mask = model.predict(image_batch)[0]

    # Threshold the predicted mask to create a binary mask
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    predicted_mask = cv2.resize(predicted_mask, (img.shape[1], img.shape[0]))

    # Overlay the predicted mask on the original image
    overlay = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    overlay[predicted_mask == 1] = [0, 255, 0]

    # Place the overlay in the right 1/3 of the canvas
    canvas[50:image["height"]+50, 2*image["width"]:, :] = overlay

    # Save the canvas as an image
    cv2.imwrite(os.path.join(os.path.dirname(__file__), 'out_ims', f'output_im_{num}.png'), canvas)

    




    

    



