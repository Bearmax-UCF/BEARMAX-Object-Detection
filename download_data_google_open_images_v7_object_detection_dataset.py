import ast
import os
import shutil
import argparse
import requests
import pandas as pd
from downloader import download_all_images

def get_num_per_class(class_ids, num_images_per_classes, train_percent_split):

    train_percent = train_percent_split
    val_percent = (1-train_percent) / 2
    test_percent = (1-train_percent) / 2

    train_images_per_class = int(train_percent * num_images_per_classes)
    val_images_per_class = int(val_percent * num_images_per_classes)
    test_images_per_class = int(test_percent * num_images_per_classes)

    dataset_subset_total = len(class_ids) * (num_images_per_classes)

    return dataset_subset_total, train_images_per_class, val_images_per_class, test_images_per_class

class DatasetClass:
    def __init__(self, id):
        self.id = id        # represent the unique class ids for each class
        self.num_train = 0
        self.num_val = 0
        self.num_test = 0

    def getNumTrain(self):
        return self.num_train

    def getNumVal(self):
        return self.num_val
    
    def getNumTest(self):
        return self.num_test

    def incrementNumTrain(self):
        self.num_train = self.num_train + 1

    def incrementNumVal(self):
        self.num_val = self.num_val + 1
    
    def incrementNumTest(self):
        self.num_test = self.num_test + 1

def process(classes, out_dir, num_images_per_classes, train_percent_split):

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

    dataset_subset_total, train_images_per_class, val_images_per_class, test_images_per_class = get_num_per_class(class_ids, num_images_per_classes, train_percent_split)

    dataset_subset_total_counter = 0

    datasetClassDict = {}

    for class_name in class_ids:
        datasetClassDict[class_name] = DatasetClass(class_name)
    
    image_list_file_list = []
    for j, url in enumerate([train_data_url, val_data_url, test_data_url]):
        if len(image_list_file_list) >= dataset_subset_total:
            break

        filename = url.split('/')[-1]
        with open(filename, 'r') as f:
            line = f.readline()
            while len(line) != 0:
                id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]

                match j:
                    case 0:
                        if class_name in datasetClassDict:
                            if datasetClassDict[class_name].getNumTrain() >= train_images_per_class:
                                break
                    case 1:
                        if class_name in datasetClassDict:
                            if datasetClassDict[class_name].getNumVal() >= val_images_per_class:
                                break
                    case 2:
                        if class_name in datasetClassDict:
                            if datasetClassDict[class_name].getNumTest() >= test_images_per_class:
                                break

                if class_name in class_ids and id not in image_list_file_list:
                    # Check total limit across all sets
                    if sum([datasetClassDict[class_name].getNumTrain(), datasetClassDict[class_name].getNumVal(), datasetClassDict[class_name].getNumTest()]) >= num_images_per_classes:
                        break
                
                    image_list_file_list.append(id)

                    dataset_subset_total_counter += 1

                    with open(image_list_file_path, 'a') as fw:
                        fw.write('{}/{}\n'.format(['train', 'validation', 'test'][j], id))

                    match j:
                        case 0:
                            datasetClassDict[class_name].incrementNumTrain()
                        case 1:
                            datasetClassDict[class_name].incrementNumVal()
                        case 2:
                            datasetClassDict[class_name].incrementNumTest()
                        
                line = f.readline()

            f.close()

    print(datasetClassDict)

    download_all_images({
        'image_list': image_list_file_path,
        'download_folder': out_dir,
        'num_processes': 5
    })

    for set_ in ['images', 'labels']:
        for dir_ in [os.path.join(out_dir, set_),
                     os.path.join(out_dir, set_, 'train'),
                     os.path.join(out_dir, set_, 'val'),
                     os.path.join(out_dir, set_, 'test')]:
            if os.path.exists(dir_):
                shutil.rmtree(dir_)
            os.makedirs(dir_)

    for j, url in enumerate([train_data_url, val_data_url, test_data_url]):
        filename = url.split('/')[-1]
        subset_ = ['train', 'val', 'test'][j]
        print(filename)
        with open(filename, 'r') as f:
            line = f.readline()
            while len(line) != 0:
                id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
                if class_name in class_ids:
                    
                    src_image_path = os.path.join(out_dir, '{}.jpg'.format(id))
                    dst_image_path = os.path.join(out_dir, 'images', subset_, '{}.jpg'.format(id))
                    dst_label_path = os.path.join(out_dir, 'labels', subset_, '{}.txt'.format(id))
    
                    # Check if image exists before processing
                    if os.path.exists(src_image_path):
                        try:
                            shutil.move(src_image_path, dst_image_path)  # Move the image
                            
                            with open(os.path.join(out_dir, 'labels', subset_, '{}.txt'.format(id)), 'a') as f_ann:
                                # class_id, xc, yx, w, h
                                x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]
                                xc = (x1 + x2) / 2
                                yc = (y1 + y2) / 2
                                w = x2 - x1
                                h = y2 - y1
        
                                f_ann.write('{} {} {} {} {}\n'.format(int(class_ids.index(class_name)), xc, yc, w, h))
                                f_ann.close()
                            
                        except FileNotFoundError:
                            print(f"File not found: {src_image_path}")

                line = f.readline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', default=[ 'Accordion',
                                               'Adhesive tape',
                                               # 'Aircraft',
                                               'Airplane',
                                               'Alarm clock',
                                               'Alpaca',
                                               'Ambulance',
                                                'Animal',
                                               'Ant',
                                               'Antelope',
                                               'Apple',
                                               'Armadillo',
                                               'Artichoke',
                                               'Auto part',
                                               # 'Axe',
                                               'Backpack',
                                               'Bagel',
                                               'Baked goods',
                                               'Balance beam',
                                               'Balloon',
                                               'Banana',
                                               'Band-aid',
                                               'Banjo',
                                               'Barge',
                                               'Barrel',
                                               'Baseball bat',
                                               'Baseball glove',
                                               'Bat (Animal)',
                                               'Bathroom accessory',
                                               'Bathroom cabinet',
                                               'Bathtub',
                                               # 'Beaker',
                                               'Bear',
                                               'Bed',
                                               'Bee',
                                               'Beehive',
                                               # 'Beer',
                                               'Beetle',
                                               'Bell pepper',
                                               'Belt',
                                               'Bench',
                                               'Bicycle',
                                               'Bicycle helmet',
                                               'Bicycle wheel',
                                               'Bidet',
                                               'Billboard',
                                               'Billiard table',
                                               'Binoculars',
                                               'Bird',
                                               'Blender',
                                               'Blue jay',
                                               'Boat',
                                               # 'Bomb',
                                               'Book',
                                               'Bookcase',
                                               'Boot',
                                               'Bottle',
                                               'Bottle opener',
                                               'Bow and arrow',
                                               'Bowl',
                                               'Bowling equipment',
                                               'Box',
                                               # 'Boy',
                                               # 'Brassiere',
                                               'Bread',
                                               'Briefcase',
                                               'Broccoli',
                                               'Bronze sculpture',
                                               'Brown bear',
                                               'Building',
                                               'Bull',
                                               'Burrito',
                                               'Bus',
                                               # 'Bust',
                                               'Butterfly',
                                               'Cabbage',
                                               'Cabinetry',
                                               'Cake',
                                               'Cake stand',
                                               'Calculator',
                                               'Camel',
                                               'Camera',
                                               'Can opener',
                                               'Canary',
                                               'Candle',
                                               'Candy',
                                               # 'Cannon',
                                               'Canoe',
                                               'Cantaloupe',
                                               'Car',
                                               'Carnivore',
                                               'Carrot',
                                               'Cart',
                                               'Castle',
                                               'Cat',
                                               'Cat furniture',
                                               'Caterpillar',
                                               'Cattle',
                                               'Ceiling fan',
                                               'Cello',
                                               'Centipede',
                                               # 'Chainsaw',
                                               'Chair',
                                               'Cheese',
                                               'Cheetah',
                                               'Chest of drawers',
                                               'Chicken',
                                               'Chime',
                                               'Chisel',
                                               'Chopsticks',
                                               'Christmas tree',
                                               'Clock',
                                               'Closet',
                                               'Clothing',
                                               'Coat',
                                               # 'Cocktail',
                                               # 'Cocktail shaker',
                                               'Coconut',
                                               'Coffee cup',
                                               'Coffee table',
                                               'Coffeemaker',
                                               'Coin',
                                               'Common fig',
                                               'Common sunflower',
                                               'Computer keyboard',
                                               'Computer monitor',
                                               'Computer mouse',
                                               'Container',
                                               'Convenience store',
                                               'Cookie',
                                               'Cooking spray',
                                               'Corded phone',
                                               'Cosmetics',
                                               'Couch',
                                               'Countertop',
                                               'Cowboy hat',
                                               'Crab',
                                               'Cream',
                                               'Cricket ball',
                                               'Crocodile',
                                               'Croissant',
                                               'Crown',
                                               'Crutch',
                                               'Cucumber',
                                               'Cupboard',
                                               'Curtain',
                                               'Cutting board',
                                               # 'Dagger',
                                               'Dairy Product',
                                               'Deer',
                                               'Desk',
                                               'Dessert',
                                               'Diaper',
                                               'Dice',
                                               'Digital clock',
                                               'Dinosaur',
                                               'Dishwasher',
                                               'Dog',
                                               'Dog bed',
                                               'Doll',
                                               'Dolphin',
                                               'Door',
                                               'Door handle',
                                               'Doughnut',
                                               'Dragonfly',
                                               'Drawer',
                                               'Dress',
                                               # 'Drill (Tool)',
                                               'Drink',
                                               'Drinking straw',
                                               'Drum',
                                               'Duck',
                                               'Dumbbell',
                                               'Eagle',
                                               'Elephant',
                                               'Envelope',
                                               'Eraser',
                                               'Face powder',
                                               'Facial tissue holder',
                                               'Falcon',
                                               'Fashion accessory',
                                               'Fast food',
                                               'Fax',
                                               'Fedora',
                                               'Filing cabinet',
                                               'Fire hydrant',
                                               'Fireplace',
                                               'Fish',
                                               'Flag',
                                               'Flashlight',
                                               'Flower',
                                               'Flowerpot',
                                               'Flute',
                                               'Flying disc',
                                               'Food',
                                               'Food processor',
                                               'Football',
                                               'Football helmet',
                                               'Footwear',
                                               'Fork',
                                               'Fountain',
                                               'Fox',
                                               'French fries',
                                               'French horn',
                                               'Frog',
                                               'Fruit',
                                               'Frying pan',
                                               'Furniture',
                                               'Garden Asparagus',
                                               'Gas stove',
                                               'Giraffe',
                                               # 'Girl',
                                               'Glasses',
                                               'Glove',
                                               'Goat',
                                               'Goggles',
                                               'Goldfish',
                                               'Golf ball',
                                               'Golf cart',
                                               'Gondola',
                                               'Goose',
                                               'Grape',
                                               'Grapefruit',
                                               'Grinder',
                                               'Guacamole',
                                               'Guitar',
                                               'Hair dryer',
                                               'Hair spray',
                                               'Hamburger',
                                               'Hammer',
                                               'Hamster',
                                               'Hand dryer',
                                               'Handbag',
                                               'Handgun',
                                               'Harbor seal',
                                               'Harmonica',
                                               'Harp',
                                               'Harpsichord',
                                               'Hat',
                                               'Headphones',
                                               'Heater',
                                               'Hedgehog',
                                               'Helicopter',
                                               'Helmet',
                                               'High heels',
                                               'Hiking equipment',
                                               'Hippopotamus',
                                               'Home appliance',
                                               'Honeycomb',
                                               'Horizontal bar',
                                               'Horse',
                                               'Hot dog',
                                               'House',
                                               'Houseplant',
                                               # 'Human arm',
                                               # 'Human body',
                                               # 'Human ear',
                                               # 'Human eye',
                                               # 'Human face',
                                               # 'Human foot',
                                               # 'Human hair',
                                               # 'Human hand',
                                               # 'Human head',
                                               # 'Human leg',
                                               # 'Human mouth',
                                               # 'Human nose',
                                               'Humidifier',
                                               'Ice cream',
                                               'Indoor rower',
                                               'Infant bed',
                                               'Insect',
                                               'Invertebrate',
                                               'Ipod',
                                               'Isopod',
                                               'Jacket',
                                               'Jacuzzi',
                                               'Jaguar (Animal)',
                                               'Jeans',
                                               'Jellyfish',
                                               'Jet ski',
                                               'Jug',
                                               'Juice',
                                               'Kangaroo',
                                               'Kettle',
                                               'Kitchen & dining room table',
                                               'Kitchen appliance',
                                               'Kitchen knife',
                                               'Kitchen utensil',
                                               'Kitchenware',
                                               'Kite',
                                               # 'Knife',
                                               'Koala',
                                               'Ladder',
                                               'Ladle',
                                               'Ladybug',
                                               'Lamp',
                                               'Land vehicle',
                                               'Lantern',
                                               'Laptop',
                                               'Lavender (Plant)',
                                               'Leopard',
                                               'Light bulb',
                                               'Light switch',
                                               'Lighthouse',
                                               'Lily',
                                               'Limousine',
                                               'Lion',
                                               'Lipstick',
                                               'Lizard',
                                               'Lobster',
                                               'Loveseat',
                                               'Luggage and bags',
                                               'Lynx',
                                               'Magpie',
                                               'Mammal',
                                               # 'Man',
                                               'Mango',
                                               'Maple',
                                               'Marine invertebrates',
                                               'Marine mammal',
                                               'Measuring cup',
                                               'Mechanical fan',
                                               'Medical equipment',
                                               'Microphone',
                                               'Microwave oven',
                                               'Milk',
                                               # 'Miniskirt',
                                                'Mirror',
                                               # 'Missile',
                                               'Mixer',
                                               'Mixing bowl',
                                               'Mobile phone',
                                               'Monkey',
                                               'Moths and butterflies',
                                               'Motorcycle',
                                               'Mouse',
                                               'Muffin',
                                               'Mug',
                                               'Mule',
                                               'Mushroom',
                                               'Musical instrument',
                                               'Musical keyboard',
                                               'Nail (Construction)',
                                               'Necklace',
                                               'Nightstand',
                                               'Oboe',
                                               'Office building',
                                               'Office supplies',
                                               'Organ (Musical Instrument)',
                                               'Ostrich',
                                               'Otter',
                                               'Oven',
                                               'Owl',
                                               'Oyster',
                                               'Paddle',
                                               'Palm tree',
                                               'Pancake',
                                               'Panda',
                                               'Paper cutter',
                                               'Paper towel',
                                               'Parachute',
                                               'Parking meter',
                                               'Parrot',
                                               'Pasta',
                                               'Pastry',
                                               'Peach',
                                               'Pear',
                                               'Pen',
                                               'Pencil case',
                                               'Pencil sharpener',
                                               'Penguin',
                                               'Perfume',
                                               'Person',
                                               'Personal care',
                                               'Personal flotation device',
                                               'Piano',
                                               'Picnic basket',
                                               'Picture frame',
                                               'Pig',
                                               'Pillow',
                                               'Pineapple',
                                               'Pitcher (Container)',
                                               'Pizza',
                                               'Pizza cutter',
                                               'Plant',
                                               'Plastic bag',
                                               'Plate',
                                               'Platter',
                                               'Plumbing fixture',
                                               'Polar bear',
                                               'Pomegranate',
                                               'Popcorn',
                                               'Porch',
                                               'Porcupine',
                                               'Poster',
                                               'Potato',
                                               'Power plugs and sockets',
                                               'Pressure cooker',
                                               'Pretzel',
                                               'Printer',
                                               'Pumpkin',
                                               'Punching bag',
                                               'Rabbit',
                                               'Raccoon',
                                               'Racket',
                                               'Radish',
                                               # 'Ratchet (Device)',
                                               'Raven',
                                               'Rays and skates',
                                               'Red panda',
                                               'Refrigerator',
                                               'Remote control',
                                               'Reptile',
                                               'Rhinoceros',
                                               # 'Rifle',
                                               'Ring binder',
                                               'Rocket',
                                               'Roller skates',
                                               'Rose',
                                               'Rugby ball',
                                               'Ruler',
                                               'Salad',
                                               'Salt and pepper shakers',
                                               'Sandal',
                                               'Sandwich',
                                               'Saucer',
                                               'Saxophone',
                                               'Scale',
                                               'Scarf',
                                               'Scissors',
                                               'Scoreboard',
                                               'Scorpion',
                                               'Screwdriver',
                                               'Sculpture',
                                               'Sea lion',
                                               'Sea turtle',
                                               'Seafood',
                                               'Seahorse',
                                               'Seat',
                                               'Segway',
                                               'Serving tray',
                                               'Sewing machine',
                                               'Shark',
                                               'Sheep',
                                               'Shelf',
                                               'Shellfish',
                                               'Shirt',
                                               'Shorts',
                                               # 'Shotgun',
                                               'Shower',
                                               'Shrimp',
                                               'Sink',
                                               'Skateboard',
                                               'Ski',
                                               'Skirt',
                                               'Skull',
                                               'Skunk',
                                               'Skyscraper',
                                               'Slow cooker',
                                               'Snack',
                                               'Snail',
                                               'Snake',
                                               'Snowboard',
                                               'Snowman',
                                               'Snowmobile',
                                               'Snowplow',
                                               'Soap dispenser',
                                               'Sock',
                                               'Sofa bed',
                                               'Sombrero',
                                               'Sparrow',
                                               'Spatula',
                                               'Spice rack',
                                               'Spider',
                                               'Spoon',
                                               'Sports equipment',
                                               'Sports uniform',
                                               'Squash (Plant)',
                                               'Squid',
                                               'Squirrel',
                                               'Stairs',
                                               'Stapler',
                                               'Starfish',
                                               'Stationary bicycle',
                                               'Stethoscope',
                                               'Stool',
                                               'Stop sign',
                                               'Strawberry',
                                               'Street light',
                                               'Stretcher',
                                               'Studio couch',
                                               'Submarine',
                                               'Submarine sandwich',
                                               'Suit',
                                               'Suitcase',
                                               'Sun hat',
                                               'Sunglasses',
                                               'Surfboard',
                                               'Sushi',
                                               'Swan',
                                               'Swim cap',
                                               'Swimming pool',
                                               'Swimwear',
                                               'Sword',
                                               'Syringe',
                                               'Table',
                                               'Table tennis racket',
                                               'Tablet computer',
                                               'Tableware',
                                               'Taco',
                                               # 'Tank',
                                               'Tap',
                                               'Tart',
                                               'Taxi',
                                               'Tea',
                                               'Teapot',
                                               'Teddy bear',
                                               'Telephone',
                                               'Television',
                                               'Tennis ball',
                                               'Tennis racket',
                                               'Tent',
                                               'Tiara',
                                               'Tick',
                                               'Tie',
                                               'Tiger',
                                               'Tin can',
                                               'Tire',
                                               'Toaster',
                                               'Toilet',
                                               'Toilet paper',
                                               'Tomato',
                                               'Tool',
                                               'Toothbrush',
                                               'Torch',
                                               'Tortoise',
                                               'Towel',
                                               'Tower',
                                               'Toy',
                                               'Traffic light',
                                               'Traffic sign',
                                               'Train',
                                               'Training bench',
                                               'Treadmill',
                                               'Tree',
                                               'Tree house',
                                               'Tripod',
                                               'Trombone',
                                               'Trousers',
                                               'Truck',
                                               'Trumpet',
                                               'Turkey',
                                               'Turtle',
                                               'Umbrella',
                                               'Unicycle',
                                               'Van',
                                               'Vase',
                                               'Vegetable',
                                               'Vehicle',
                                               'Vehicle registration plate',
                                               'Violin',
                                               'Volleyball (Ball)',
                                               'Waffle',
                                               'Waffle iron',
                                               'Wall clock',
                                               'Wardrobe',
                                               'Washing machine',
                                               'Waste container',
                                               'Watch',
                                               'Watercraft',
                                               'Watermelon',
                                               # 'Weapon',
                                               'Whale',
                                               'Wheel',
                                               'Wheelchair',
                                               'Whisk',
                                               'Whiteboard',
                                               'Willow',
                                               'Window',
                                               'Window blind',
                                               # 'Wine',
                                               'Wine glass',
                                               # 'Wine rack',
                                               'Winter melon',
                                               'Wok',
                                               # 'Woman',
                                               'Wood-burning stove',
                                               'Woodpecker',
                                               'Worm',
                                               'Wrench',
                                               'Zebra',
                                               'Zucchini'
                                            ])
    parser.add_argument('--out-dir', default='./data')
    parser.add_argument('--num_images_per_classes', default=30000, type=int, help='Max number of images per class.')
    parser.add_argument('--train_percent_split', default=0.7, type=int, help='Percentage of the split of the dataset that is in the training set.')
    args = parser.parse_args()

    classes = args.classes
    if type(classes) is str:
        classes = ast.literal_eval(classes)

    out_dir = args.out_dir

    num_images_per_classes = args.num_images_per_classes

    train_percent_split = args.train_percent_split

    process(classes, out_dir, num_images_per_classes, train_percent_split)
