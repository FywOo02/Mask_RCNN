import cv2
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.visualize import display_instances
class FacialFeaturesConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "facial features"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + balloon

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.85


image = cv2.imread('images/train/i001qd-mn.jpg')

config = FacialFeaturesConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir='logs')
model.load_weights('logs/facial features20231229T1746/mask_rcnn_facial features_0020.h5', by_name=True)

result = model.detect([image])

# print(result[0])
class_names = ['BG','eye','eyebrow','ear','nose','mouse']
display_instances(image,result[0]['rois'], result[0]['masks'], result[0]['class_ids'],class_names,scores=result[0]['scores'], title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None)