import cv2
import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage

from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = 'mrcnn_tutorial'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class FingerprintsConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Fingerprints"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + fingerprint

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 704 
    IMAGE_MAX_DIM = 832   
    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([143.62647976011993,143.62647976011993,143.62647976011993])

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 700

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 150

    BACKBONE = 'resnet101'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # TRAIN_ROIS_PER_IMAGE = 32
    # MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000
    
    # dalsie parametre na zvazenie
    # BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # RPN_ANCHOR_STRIDE = 1
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # TRAIN_ROIS_PER_IMAGE = 200
    # MAX_GT_INSTANCES = 100
    # DETECTION_MAX_INSTANCES = 100


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        print()
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


dataset_test = CocoLikeDataset()
dataset_test.load_data('./data/test/annotations.json', './data/')
dataset_test.prepare()


class InferenceConfig(FingerprintsConfig):
    DETECTION_MIN_CONFIDENCE = 0.01
    # DETECTION_MAX_INSTANCES = 100


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config= FingerprintsConfig(),
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = "mask_rcnn_fingerprints_0046.h5"

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

img_ids = dataset_test._image_ids
iou_total = 0
total_masks = 0
low_iou = 100
best_iou = 0
for i in img_ids:
    img = dataset_test.load_image(i)
    results = model.detect([img], verbose=1)
    r = results[0]
    masks = r['masks']
    mask_ground, class_ids = dataset_test.load_mask(i)
    iou = utils.compute_overlaps_masks(masks, mask_ground)
    total_masks += len(iou)
    iou_total += iou.sum()


    print("in proccess  iou: ", iou_total)
    print("in proccess total masks: ", total_masks)
print("IOU = ", iou_total/total_masks)
