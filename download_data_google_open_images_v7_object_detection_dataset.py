import ast
import os
import shutil
import argparse

import requests

import pandas as pd


def process(classes, data_out_dir, yolov8_format):

    train_data_url = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
    val_data_url = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'
    test_data_url = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'

    downloader_url = 'https://raw.githubusercontent.com/openimages/dataset/master/downloader.py'

    class_names_all_url = 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv'

    for url in [train_data_url, val_data_url, test_data_url, class_names_all_url, downloader_url]:
        if not os.path.exists(url.split('/')[-1]):
            r = requests.get(url)
            with open(url.split('/')[-1], 'wb') as f:
                f.write(r.content)

    class_ids = []

    classes_all = pd.read_csv(class_names_all_url.split('/')[-1])

    for class_ in classes:
        if class_ not in list(classes_all['DisplayName']) or class_ not in list(classes_all['DisplayName']):
            raise Exception('Class name not found: {}'.format(class_))
        class_index = list(classes_all['DisplayName']).index(class_)
        class_ids.append(classes_all['LabelName'].iloc[class_index])

    image_list_file_path = os.path.join('.', 'image_list_file')

    image_list_file_list = []
    for j, url in enumerate([train_data_url, val_data_url, test_data_url]):
        filename = url.split('/')[-1]
        with open(filename, 'r') as f:
            line = f.readline()
            while len(line) != 0:
                id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
                if class_name in class_ids and id not in image_list_file_list:
                    image_list_file_list.append(id)
                    with open(image_list_file_path, 'a') as fw:
                        fw.write('{}/{}\n'.format(['train', 'validation', 'test'][j], id))
                line = f.readline()

            f.close()

    out_dir = './.out'
    shutil.rmtree(out_dir, ignore_errors=True)
    os.system('python downloader.py {} --download_folder={}'.format(image_list_file_path, out_dir))

    DATA_ALL_DIR = out_dir

    for set_ in ['train', 'val', 'test']:
        for dir_ in [os.path.join(data_out_dir, set_),
                     os.path.join(data_out_dir, set_, 'imgs'),
                     os.path.join(data_out_dir, set_, 'anns')]:
            if os.path.exists(dir_):
                shutil.rmtree(dir_)
            os.makedirs(dir_)

    for j, url in enumerate([train_data_url, val_data_url, test_data_url]):
        filename = url.split('/')[-1]
        set_ = ['train', 'val', 'test'][j]
        print(filename)
        with open(filename, 'r') as f:
            line = f.readline()
            while len(line) != 0:
                id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
                if class_name in class_ids:
                    if not os.path.exists(os.path.join(data_out_dir, set_, 'imgs', '{}.jpg'.format(id))):
                        shutil.copy(os.path.join(DATA_ALL_DIR, '{}.jpg'.format(id)),
                                    os.path.join(data_out_dir, set_, 'imgs', '{}.jpg'.format(id)))
                    with open(os.path.join(data_out_dir, set_, 'anns', '{}.txt'.format(id)), 'a') as f_ann:
                        # class_id, xc, yx, w, h
                        x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]
                        xc = (x1 + x2) / 2
                        yc = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1

                        f_ann.write('{} {} {} {} {}\n'.format(int(class_ids.index(class_name)), xc, yc, w, h))
                        f_ann.close()

                line = f.readline()

    shutil.rmtree(out_dir, ignore_errors=True)

    if yolov8_format:
        for set_ in ['train', 'val', 'test']:
            for dir_ in [os.path.join(data_out_dir, 'images', set_),
                         os.path.join(data_out_dir, 'labels', set_)]:
                if os.path.exists(dir_):
                    shutil.rmtree(dir_)
                os.makedirs(dir_)

            for filename in os.listdir(os.path.join(data_out_dir, set_, 'imgs')):
                shutil.copy(os.path.join(data_out_dir, set_, 'imgs', filename), os.path.join(data_out_dir, 'images', set_, filename))
            for filename in os.listdir(os.path.join(data_out_dir, set_, 'anns')):
                shutil.copy(os.path.join(data_out_dir, set_, 'anns', filename), os.path.join(data_out_dir, 'labels', set_, filename))

            shutil.rmtree(os.path.join(data_out_dir, set_))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', default=['Person', # 'Clothing', 'Book', 'Office supplies', 'Footwear', 'Window', 'Box', 'Bottle', 'Food', 'Door',
                                              # 'Flower', 'Wheel', 'Plant', 'Car', 'Human hair', 'Human arm', 'Human head', 'Building', 
                                              # 'Human body', 'Mammal', 'House', 'Chair', 'Tire', 'Fashion accessory', 'Table',
                                              # 'Skyscraper', 'Land vehicle', 'Boat', 'Jeans', 'Human eye', 'Human hand', 'Human leg', 'Toy',
                                              # 'Tower', 'Human nose', 'Bicycle wheel', 'Glasses', 'Dress', 'Vehicle', 'Bird', 'Sports equipment',
                                              # 'Human mouth', 'Palm tree', 'Tableware', 'Drink', 'Bicycle', 'Furniture', 'Snack', 
                                              # 'Sculpture', 'Flag', 'Dog', 'Dessert', 'Microphone', 'Fruit', 'Jacket', 'Guitar', 'Fast food', 
                                              # 'Drum', 'Sunglasses', 'Poster', 'Fish', 'Baked goods', 'Shelf', 'Houseplant', 'Flowerpot', 'Airplane', 
                                              # 'Sports uniform', 'Vegetable', 'Human ear', 'Animal', 'Shorts', 'Musical instrument', 'Helmet', 
                                              # 'Bicycle helmet', 'Duck', 'Wine', 'Cat', 'Auto part', 'Balloon', 'Motorcycle', 'Horse', 'Hat', 'Train', 
                                              # 'Wine glass', 'Truck', 'Rose', 'Picture frame', 'Bus', 'Football helmet', 'Desk', 'Cattle', 'Bee', 'Tie', 
                                              # 'Hiking equipment', 'Butterfly', 'Swimwear', 'Billboard', 'Goggles', 'Beer', 'Laptop', 'Cabinetry', 
                                              # 'Marine invertebrates', 'Insect', 'Trousers', 'Goose', 'Strawberry', 'Vehicle registration plate', 
                                              # 'Van', 'Shirt', 'Traffic light', 'Bench', 'Umbrella', 'Sun hat', 'Paddle', 'Tent', 'Sunflower', 
                                              # 'Coat', 'Doll', 'Camera', 'Mobile phone', 'Tomato', 'Pumpkin', 'Tree', 'Human face',
                                              # 'Traffic sign', 'Computer monitor', 'Stairs', 'Candle', 'Pastry', 'Cake', 'Roller skates', 'Lantern', 
                                              # 'Plate', 'Coffee cup', 'Coffee table', 'Bookcase', 'Watercraft', 'Football', 'Office building', 
                                              # 'Maple', 'Curtain', 'Kitchen appliance', 'Muffin', 'Canoe', 'Computer keyboard', 'Swan', 'Bowl', 'Mushroom', 
                                              # 'Cocktail', 'Drawer', 'Castle', 'Couch', 'Christmas tree', 'Taxi', 'Penguin', 'Cookie', 'Apple', 
                                              # 'Swimming pool', 'Deer', 'Porch', 'Bread', 'Bowling equipment', 'Television', 'Fountain', 
                                              # 'Lamp', 'Fedora', 'Bed', 'Beetle', 'Pillow', 'Ski', 'Carnivore', 'Platter', 'Sheep', 'Elephant', 
                                              # 'Boot', 'High heels', 'Countertop', 'Salad', 'Cowboy hat', 'Seafood', 'Chicken', 'Coin', 'Monkey', 'Helicopter', 
                                              # 'Tin can', 'Sandal', 'Juice'
                                              ])
    parser.add_argument('--out-dir', default='./data')
    parser.add_argument('--yolov8-format', default=True)
    args = parser.parse_args()

    classes = args.classes
    if type(classes) is str:
        classes = ast.literal_eval(classes)

    out_dir = args.out_dir

    yolov8_format = True if args.yolov8_format in ['T', 'True', 1, '1'] else False

    process(classes, out_dir, yolov8_format)
