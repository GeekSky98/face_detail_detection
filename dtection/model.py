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
sys.path.append('C:\\Users\\whgks\\PycharmProjects\\face_detail_detection\\dtection')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tarfile
import os
from matplotlib.patches import Rectangle
from utils import load_image_into_numpy_array, get_model_detection_function, generate_xywh
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

def category():
    label_map_path = 'dtection/models/research/object_detection/data/mscoco_label_map.pbtxt'
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    return category_index, label_map_dict

category_index, label_map_dict = category()

model_name = 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8'
image_path = './dtection/models/research/object_detection/test_images/image2.jpg'

def tensorflow_api(model_name, first=False):
    if first:
        tar_path = os.path.join('./dtection/zoo/', model_name + '.tar.gz')
        tar_unzip = tarfile.open(tar_path)
        tar_unzip.extractall('./dtection/zoo')

    pipeline_config = os.path.join('dtection/models/research/object_detection/configs/tf2/', model_name + '.config')
    model_dir = 'dtection/models/research/object_detection/test_data/checkpoint/'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']

    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    model = get_model_detection_function(detection_model)

    return model

od_model = tensorflow_api(model_name)

def predict(model, image_path, threshold=0.3, label_id_offset=1):
    image = load_image_into_numpy_array(image_path)
    image_for_pred = image

    image_for_pred = tf.cast(tf.expand_dims(image_for_pred, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = model(image_for_pred)

    person_idx = tf.where(detections['detection_classes'][0] == 0)
    score_idx = tf.where(detections['detection_scores'][0] > threshold)
    person_idx_set = set(np.reshape(person_idx, [-1]))
    score_idx_set = set(np.reshape(score_idx, [-1]))

    pred_box = tf.gather(detections['detection_boxes'][0], list(person_idx_set & score_idx_set))
    # pred_class = tf.gather(detections['detection_classes'][0], list(person_idx_set & score_idx_set))
    # pred_score = tf.gather(detections['detection_scores'][0], list(person_idx_set & score_idx_set))

    x, y, w, h = generate_xywh(image, pred_box)

    ax = plt.subplot()
    for idx in range(len(pred_box)):
        rect = Rectangle((x[idx], y[idx]), w[idx], h[idx], fill=False, color='red')
        ax.add_patch(rect)
        ax.text(x[idx], y[idx], str(idx+1)+'option')
    plt.imshow(image)
    plt.savefig('./dtection/test')
    plt.clf()

    return x, y, w, h

x, y, w, h = predict(od_model, image_path)


def image_crop(image, option=1):
