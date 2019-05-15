import os
import sys
import random
import math
import re
import csv
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from libtiff import TIFF
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import SpaceNet_Utils.python.spaceNetUtilities.geoTools as gT


#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448
    IMAGE_SHAPE = [448, 448, 3]

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = ShapesConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class SpaceNetDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def get_obj_index(self, image):
        n = np.max(image)
        return n
    def load_shapes(self, img_list, aug=True, rotate = False):#, img_floder, mask_floder,json_floder, img16bit_floder):
        """Generate the requested number of synthetic images.
        #count: number of images to generate.
        image_id_list : list of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "shapes")
        k = 0 
        for i in img_list:
            pattern = re.compile(r'AOI_._\w*_')
            keyname = re.findall(pattern, i)
            json_floder = "/home/ly/data/dl_data/spacenet/%s_Train/geojson/buildings"%keyname
            img16bit_floder = "/home/ly/data/dl_data/spacenet/%s_Train/RGB-PanSharpen"%keyname
            img_floder = "/home/ly/data/dl_data/spacenet/%s_Train/RGB_8bit"%keyname
            mask_floder = "/home/ly/data/dl_data/spacenet/%s_Train/mask"%keyname

            image_id = keyname + i.split("_")[-1][3:-4]
            mask_path = mask_floder + "/mask_RGB-" + i.split("-")[-1]
            json_path = json_floder + "/buildings_%s_img"%keyname + i.split("_")[-1][3:-4] +".geojson"
            img_path = img_floder + "/" + i
            img16bit_path = img16bit_floder + "/" + i
            cv_img = cv2.imread(img_path)
            self.add_image("shapes", image_id=k, path= img_path,
                           width=cv_img.shape[1], height=cv_img.shape[0],rotate = rotate,
                           mask_path= mask_path, json_path= json_path, img16bit_path= img16bit_path)
            k += 1
        if aug:
            augdata_floder = "/home/ly/data/dl_data/spacenet/AOI_4_Shanghai_Train/augdata"
            augdata_list = os.listdir(augdata_floder)
            random.shuffle(augdata_list)
            for j in augdata_list:
                json_floder = "/home/ly/data/dl_data/spacenet/AOI_4_Shanghai_Train/geojson/buildings"
                img16bit_floder = "/home/ly/data/dl_data/spacenet/AOI_4_Shanghai_Train/RGB-PanSharpen"
                img_floder = "/home/ly/data/dl_data/spacenet/AOI_4_Shanghai_Train/RGB_8bit"
                mask_floder = "/home/ly/data/dl_data/spacenet/AOI_4_Shanghai_Train/mask"
                augmask_floder = "/home/ly/data/dl_data/spacenet/AOI_4_Shanghai_Train/augmask"
                prename = j[4:]
                if j[0:3] == "sat" or "brt":
                    mask_path = mask_floder + "/mask_" + prename
                if j[0:3] == "rot":
                    mask_path = augmask_floder + "/rot_" + prename
                    rotate = True
                json_path = json_floder + "/buildings_AOI_4_Shanghai_Train_img" + prename[14:]+".geojson"
                img_path = os.path.join(img_floder, prename)
                img16bit_path = os.path.join(img16bit_floder, prename)
                cv_img = cv2.imread(img_path)
                self.add_image("shapes", image_id=k, path=img_path,
                               width=cv_img.shape[1], height=cv_img.shape[0], rotate=rotate,
                               mask_path=mask_path, json_path=json_path, img16bit_path=img16bit_path)
                k += 1
        print("the number of data is ", k)

    def getcoords_from_json(self, image_id):
        jsonpath = self.image_info[image_id]['json_path']
        img16bit = self.image_info[image_id]['img16bit_path']
        buildingList = gT.convert_wgs84geojson_to_pixgeojson(jsonpath, img16bit, pixPrecision=2)
        p1 = re.compile(r'[(](.*)[)]', re.S)   #贪婪匹配  ( )) 匹配最外边两个
        p2 = re.compile(r'[(](.*?)[)]', re.S)  #最小匹配
        pols = []
        for build in buildingList:
            polygon = str(build['polyPix'])
            #print(polygon)
            coords = re.findall(p1,polygon)
            #print(coords)
            coords2 = re.findall(p2, str(coords))   #未解决中央空岛
            #print(coords2)
            points = coords2[0].split(",")
            pol_array = []
            for point in points:
                x, y, z = point.split(" ")
                pol_array.append([int(float(x)), int(float(y))])
            pols.append(pol_array)
        return pols   
            
    def draw_mask(self, image_id):
        #print("draw_mask-->",image_id)
        #print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        try:
            pols = self.getcoords_from_json(image_id)
        except:
            mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
            label = ["0"]
            label = np.array(label)
            return mask, label
        else:
            num_obj = len(pols)
            mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
            label = []
            for index in range(num_obj):
                pol_array = np.array([pols[index]])
                im = np.zeros([info['height'], info['width']], dtype=np.uint8)
                cv2.fillPoly(im, pol_array,1)
                if info['rotate']:
                    im = Image.fromarray(im)
                    im.rotate(90, Image.BICUBIC)
                mask[:, :, index] = np.array(im)
                label.append("1")
            label = np.array(label)
            return mask, label

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        #print("image_id",image_id)
        info = self.image_info[image_id]
        #mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
        mask, class_ids = self.draw_mask(image_id )
        return mask, class_ids.astype(np.int32)
    
    def load_unetmask(self, image_id):
        info = self.image_info[image_id]
        unet_mask = Image.open(info['mask_path'])
        unet_mask = unet_mask.resize((448, 448))
        unet_mask = np.array(unet_mask).reshape((448, 448, 1))
        return unet_mask

# Training dataset
'''
img4_dir= "/home/ly/data/dl_data/spacenet/AOI_4_Shanghai_Train/RGB_8bit"
img5_dir= "/home/ly/data/dl_data/spacenet/AOI_5_Khartoum_Train/RGB_8bit"
img4list = os.listdir(img4_dir)
img5list = os.listdir(img5_dir)
imglist = img4list + img5list
random.shuffle(imglist)
print(len(imglist)) 
# 保存列表，以防不测

with open("trainlist.txt", "w") as f:
    for i in imglist:
        f.write(i)
        f.write("\n")
'''
imglist= []
with open("trainlist.txt","r") as f:
    for i in f.readlines():
        imglist.append(i.strip())

dataset_train = SpaceNetDataset()
dataset_train.load_shapes(imglist[:-400], aug=False)#, img_dir, mask_dir,json_dir, img16bit_dir)
dataset_train.prepare()

# Validation dataset
dataset_val = SpaceNetDataset()
dataset_val.load_shapes(imglist[-400:], aug=False)#, img_dir, mask_dir,json_dir, img16bit_dir)
dataset_val.prepare()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

#weight_path = "/home/ly/data/pycharm/Mask_RCNN-master/logs/shapes20180813T2230/mask_rcnn_shapes_0140.h5"
#model.load_weights(weight_path, by_name=True)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE * 10,
            epochs=75,
            layers='heads')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE * 5,
            epochs=120,
            layers="all")
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE * 2,
            epochs=250,
            layers='maskrcnn')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=300,
            layers='unet')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 2,
            epochs=350,
            layers="unet+")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 5,
            epochs=400,
            layers="all")

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=500,
            layers="all")

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=600,
            layers="all")