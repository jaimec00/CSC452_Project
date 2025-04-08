from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib.patches import Polygon

def main():
    # train, val or testing

    # paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(data_dir, "images/images")
    img_test_dir = os.path.join(img_dir, "livecell_test_images")
    annotations_dir = os.path.join(data_dir, "annotations")

    # just load testing for now
    annFile = os.path.join(annotations_dir, "testing.json")
    coco = COCO(annFile)

    imgIds = coco.getImgIds()  
    img = coco.loadImgs(imgIds[0])[0]
    img_file = img["file_name"]
    img_class = img_file.split("_")[0]
    img_path = os.path.join(img_test_dir, f"{img_class}/{img_file}")

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
    anns = coco.loadAnns(annIds)

    image = mpimg.imread(img_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')

    # Draw each bounding box on the image
    for ann in anns:

        segmentation = ann["segmentation"]
        coords = segmentation_2_coords(segmentation)
        ax = plt.gca()
        polygon_patch = Polygon(coords, closed=True, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(polygon_patch)

        # if you want to do the bounding boxes instead
        # bbox = ann['bbox']
        # x, y, width, height = bbox
        # plt.gca().add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor='red', linewidth=2))
        
    plt.show()

def segmentation_2_coords(segmentation):
    '''converts flattened xy coords into list of list, each inner list being an x,y pair'''
    segmentation = segmentation[0]
    coords = [[segmentation[i], segmentation[i+1]] for i in range(0,len(segmentation),2)]

    return coords


if __name__ == "__main__":
    main()