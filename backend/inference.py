import argparse
import tensorflow as tf
import numpy as np
from typing import TypeVar, List, Tuple, Union
# from tensorflow.keras.applications

from config import CONFIG

Tensor = TypeVar("Tensor")

#======== HELPER FUNCTION ================================================
def decode_img(img: str, png=True) -> Tensor:
    """Convert the compressed string into a 3d uint8 tensor
    Args:
        img (str): path to image
        png (bool, optional): whether image is of type png or tfif. Defaults to True.

    Returns:
        Tensor: image loaded in as a tensor
    """
    if png:
        img = tf.image.decode_png(img, channels=3)
    else:
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.keras.applications.resnet50.preprocess_input(img)
    return tf.image.resize(img, [224, 224])

def process_path(file_path: str) -> Tensor:
    """Load image in as a tensor

    Args:
        file_path ([str]): path to image

    Returns:
        Tensor: tensor of image
    """
    ext = file_path.split(".")[-1]
    img = tf.io.read_file(file_path)
    if ext == "png":
        img = decode_img(img)
    else:
        img = decode_img(img, png=False)
    return img

def preprocess_img(img_input: Union[str, np.ndarray]) -> Tensor:
    """Process image to feed in mobilenetv2

    Args:
        img_input (Union[str, np.ndarray]): path to image or numpy array of image

    Returns:
        Tensor: tensor of image
    """
    # TODO: I need to refactor this part of the code
    if isinstance(img_input, np.ndarray):
        img = tf.image.resize(img_input, [224, 224])
    else:
        img = process_path(img_input)
    # TODO: Ensure that the app handles file upload and drag or url
    img = tf.reshape(img, [1, 224, 224, 3])
    return tf.image.convert_image_dtype(img, dtype=tf.float32)

Serialized_Model = TypeVar("Serialized Model")

def make_prediction(model: Serialized_Model, img_path: Union[str, np.ndarray]) -> str:
    """Make predictions using pretrained model

    Args:
        model (Serialized_Model): pretrained model
        img_path (Union[str, np.ndarray]): path to image

    Returns:
        str: prediction (cat or dog)
    """
    img = preprocess_img(img_path)
    class_names = ["cat", "dog"]
    pred = model.predict(img).flatten()
    pred = tf.nn.sigmoid(pred)
    pred = tf.where(pred < 0.5, 0, 1)
    pred = class_names[pred.numpy()[0]]
    return pred


def get_args():
    parser = argparse.ArgumentParser(
        description="Cat_vs_Dog_clf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--file", required=True, type=str,
                    help="file for making prediction")

    return parser.parse_args()

#=====================================================================================


if __name__ == "__main__":
    # model = tf.keras.models.load_model(CONFIG.models / "content/cat_vs_dog/")
    # args = get_args()
    # print(f"Prediction: {make_prediction(model, args.file)}")
    pass
