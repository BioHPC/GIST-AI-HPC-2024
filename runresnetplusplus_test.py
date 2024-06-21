import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from data_generator import DataGen
from unet import Unet
from resunet import ResUnet
from m_resunet import ResUnetPlusPlus
from metrics import dice_coef, dice_loss

import time

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # choose gpu
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

start_time = tf.timestamp()

if __name__ == "__main__":
    file_path = "files/"
    model_path = "files/resunetplusplus_kvasir-SEG_epoch100.h5"

    try:
        os.mkdir("files")
    except:
        pass

    test_path = "data/Kvasir-SEG/test/"

    # testing
    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "masks", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()

    image_size = 256
    batch_size = 8
    lr = 1e-4

    test_steps = len(test_image_paths) // batch_size

    test_gen = DataGen(image_size, test_image_paths, test_mask_paths, batch_size=batch_size)

    model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef, "MeanIoU": MeanIoU, "Precision": Precision, "Recall": Recall})

    results = model.evaluate(test_gen, steps=test_steps)
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")

end_time = tf.timestamp()
final_time = end_time - start_time
print(f"Total time taken: {final_time} seconds")
