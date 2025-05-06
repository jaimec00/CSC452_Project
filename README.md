CSC452 project

Bacterial Cell Segmentation AI Model using a U-Net Architecture

The data used was from LiveCell Dataset (https://sartorius-research.github.io/LIVECell/), the annotations and images used can be downloaded from this link, but we recommend running the data/get_data.py script which streamlines the process:

    full zip file of images: http://livecell-dataset.s3.eu-central-1.amazonaws.com/
    testing annotations: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json
    training annotations: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json
    validation annotations: http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json


To download the data, train the model, and test it, you can upload the train_colab.py notebook to google colab. It fetches the repository, download and cleans the data, and trains the model. We also include the model parameters in trained_model/output.zip as model.keras, along with a few example predictions from our model.
