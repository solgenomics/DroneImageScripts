
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
import cv2
import csv
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
import matplotlib.backends.backend_pdf

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log_file_path", required=False, help="file path to write log to. useful for using from the web interface")
ap.add_argument("-i", "--input_annotations_dir", required=True, help="file directory holding all annotation xml files")
ap.add_argument("-p", "--model_dir", required=True, help="dir where to save checkpoints")
ap.add_argument("-o", "--model_path", required=True, help="file where to save trained model")
ap.add_argument("-e", "--outfile_annotated", required=True, help="file path where the annotated image pdf saved")
ap.add_argument("-a", "--results_outfile", required=True, help="file path where the boxes are saved")

args = vars(ap.parse_args())

log_file_path = args["log_file_path"]
input_annotations_dir = args["input_annotations_dir"]
model_dir = args["model_dir"]
model_path = args["model_path"]
outfile_annotated = args["outfile_annotated"]
results_outfile = args["results_outfile"]

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

# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "plotimage_cfg"
    # number of classes (background + plot image)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.subplot(n_images, 2, i*2+1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        pyplot.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    # show the figure
    pyplot.show()

test_set = PlotImageDataset()
test_set.load_dataset(input_annotations_dir, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir=model_dir, config=cfg)
# load model weights
model.load_weights(model_path, by_name=True)
# plot predictions for test dataset
# plot_actual_vs_predicted(test_set, model, cfg)

out_figures = []

i = 0
image = test_set.load_image(i)
# mask, _ = test_set.load_mask(i)
# convert pixel values (e.g. center)
scaled_image = mold_image(image, cfg)
# convert image into one sample
sample = expand_dims(scaled_image, 0)
# make prediction
yhat = model.detect(sample, verbose=0)[0]
print(yhat)

for box in yhat['rois']:
    y1, x1, y2, x2 = box
    width, height = x2 - x1, y2 - y1
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)

pyplot.figure()
pyplot.imshow(image)
pyplot.title('Predicted')
fig = pyplot.gcf()

# show the figure
out_figures.append(fig)

pdf = matplotlib.backends.backend_pdf.PdfPages(outfile_annotated)
for fig in out_figures:
    pdf.savefig(fig)
pdf.close()

with open(results_outfile, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(yhat['rois'])
writeFile.close()
