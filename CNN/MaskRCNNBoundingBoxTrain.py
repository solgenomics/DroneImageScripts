#python3 /home/production/cxgn/DroneImageScripts/CNN/MaskRCNNBoundingBoxTrain.py --input_annotations_dir '/home/production/cxgn/sgn//static/documents/tempfiles/drone_imagery_keras_cnn_maskrcnn_dir'  --log_file_path '/var/log/sgn/error.log'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from matplotlib import pyplot

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_annotations_dir", required=True, help="file directory holding all annotation xml files")
ap.add_argument("-p", "--output_model_dir", required=True, help="dir where to save checkpoints")
ap.add_argument("-o", "--output_model_path", required=True, help="file where to save trained model")

args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_annotations_dir = args["input_annotations_dir"]
output_model_dir = args["output_model_dir"]
output_model_path = args["output_model_path"]

if sys.version_info[0] < 3:
    raise Exception("Must use Python3. Use python3 in your command line.")

if log_file_path is not None:
    sys.stderr = open(log_file_path, 'a')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class PlotImageDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, annotations_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "plotimage")
        # define data locations
        # images_dir = dataset_dir + '/images/'
        # annotations_dir = dataset_dir + '/annots/'
        # find all images
        counter = 0
        annotations = listdir(annotations_dir)
        for filename in annotations:
            ann_path = annotations_dir + '/' + filename
            eprint(ann_path)
            tree = ElementTree.parse(ann_path)
            root = tree.getroot()
            image_id = int(root.find('image_id').text)
            # extract image id
            # image_id = filename[:-4]
            # skip bad images
            # if image_id in ['00090']:
                # continue
            # skip all images after 150 if we are building the train set
            if is_train and counter >= round(len(annotations)/5):
                counter += 1
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and counter < round(len(annotations)/5):
                counter += 1
                continue
            #img_path = images_dir + filename
            img_path = root.find('image_path').text
            #ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
            counter += 1

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('plotimage'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# define a configuration for the model
class PlotImageConfig(Config):
    # define the name of the configuration
    NAME = "plotimage_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131

# prepare train set
train_set = PlotImageDataset()
train_set.load_dataset(input_annotations_dir, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = PlotImageDataset()
test_set.load_dataset(input_annotations_dir, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = PlotImageConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir=output_model_dir, config=config)
# load weights (mscoco) and exclude the output layers
# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model.load_weights(model.get_imagenet_weights(), by_name=True)
# train weights (output layers or 'heads' or 'all')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='all')
model.keras_model.save_weights(output_model_path)
