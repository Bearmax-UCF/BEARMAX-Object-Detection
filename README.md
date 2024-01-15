# BEARMAX-Object-Detection
Module for object detection on Bearmax with the Yolov8 pretrained model and Open Images v7 dataset

## How to install requirements
- First install a virtual python environment using the following script

```bash
python3 -m venv venv
```
- Then activate the virtual python environment with the following script:

```bash
source ~/venv/bin/activate
```

- Then install the dependencies in requirements.txt with the script below

```bash
pip install -r requirements.txt
```

- To exit the virtual environment use type in the script below

```bash
deactivate
```

## Information about download_data_google_open_images_v7_object_detection_dataset.py
- download_data_google_open_images_v7_object_detection_dataset.py file is to download the Open Images v7 dataset. 
    - The classes in this file can be edited depending on which classes you want to download. For the class names, you can refer to the OpenImagesV7.yaml file in the following link: https://docs.ultralytics.com/datasets/detect/open-images-v7/#dataset-yaml.
        - The more classes you want, the longer the download.
    - The outputs of this file include .csv files with training and validation annotations, the downloader.py file (which won't be necessary to use later), and eventually the data folder with the images and labels for the train, test, and validation sets. The data folder is formatted to use the Yolov8 model for object detection. More information about the model can be found on the Yolov8 Github or the Ultralytics website. 
        - https://github.com/ultralytics/ultralytics
        - https://docs.ultralytics.com/models/yolov8/ 

## Information about config.yaml
- config.yaml is the variables necessary for the train_and_val.py program. 
    - Modify the path variable to the location of where the data folder will be on your device.
    - Modify the class names to match the classes that were downloaded from the Open Images v7 dataset in the download_data_google_open_images_v7_object_detection_dataset.py file.

## Information about train_and_val.py
- train_and_val.py is the code to run the training and validation of the pretrained Yolov8 model. In order to utilize the model for real-time object detection, make sure to save the model as it will be trained with the classes you want to detect with the webcam.

## Information about openimagesv7.names
- openimagesv7.names is meant to have all the names of the objects that could be viewed and labeled via real-time object detection with the webcam. 
    - This file can have the classes edited but ```__Background__``` stays in the default.
    - Modify the class names to match the classes that were downloaded from the Open Images v7 dataset in the download_data_google_open_images_v7_object_detection_dataset.py file.

## Information about detector.py
- detector.py contains the class to handle the loading of the saved model after the training and validation as well as the creation of the bounding boxes in real-time.

## Information about main.py
- main.py has the code that you run to start the webcam and real-time object detection. Feel free to put various objects in front of the camera or move the camera to face around the surrounding area. 
    - Just remember that the objects that it detects will be based on the classes that the Yolov8 model is trained on as well as the classes listed in the openimagesv7.names file.