import numpy as np
from six import BytesIO
from PIL import Image

import tensorflow as tf


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
  def detect_fn(image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn


def swap_xy(boxes):
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1
    )


def generate_xywh(image, box):
    height, width = image.shape[:2]

    box_swap = swap_xy(box)
    box_xywh = convert_to_xywh(box_swap)

    box_w = box_xywh[:,2].numpy() * width
    box_h = box_xywh[:,3].numpy() * height
    box_x = ((box_xywh[:, 0].numpy()) * width) - (box_w / 2)
    box_y = ((box_xywh[:, 1].numpy()) * height) - (box_h / 2)

    return box_x, box_y, box_w, box_h


def get_corners(coordinates):
    left = coordinates[0]
    top = coordinates[1]
    right = left + coordinates[2]
    bottom = top + coordinates[3]

    return left, top, right, bottom
