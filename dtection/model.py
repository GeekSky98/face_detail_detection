# Install tensorflow object detection api
'''
%cd detection
!git clone https://github.com/tensorflow/models
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
'''
import sys
sys.path.append('./utils.py')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tarfile
import os
from utils import load_image_into_numpy_array, get_model_detection_function
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


model_name = 'faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8'

tar_unzip = tarfile.open('./zoo/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz')
tar_unzip.extractall('./zoo')

pipeline_config = os.path.join('models/research/object_detection/configs/tf2/', model_name + '.config')
model_dir = 'models/research/object_detection/test_data/checkpoint/'

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

detect_fn = get_model_detection_function(detection_model)

label_map_path = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

image_dir = 'models/research/object_detection/test_images/'
image_path = os.path.join(image_dir, 'image1.jpg')
image = load_image_into_numpy_array(image_path)

input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
result_image = image

viz_utils.visualize_boxes_and_labels_on_image_array(
      result_image,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.70,
      agnostic_mode=False
      )

plt.imshow(result_image)
plt.savefig('./test')
plt.show()