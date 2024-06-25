import tensorflow as tf
import os
import numpy as np
from glob import glob
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from m_resunet import ResUnetPlusPlus
from metrics import dice_coef, dice_loss
import argparse
import time
import nvtx

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

start_time = tf.timestamp()

def parse_image(img_path, image_size):
    image_rgb = tf.io.read_file(img_path)
    image_rgb = tf.image.decode_jpeg(image_rgb, channels=3)
    image_rgb = tf.image.resize(image_rgb, [image_size, image_size])
    image_rgb = tf.cast(image_rgb, tf.float32) / 255.0
    return image_rgb

def parse_mask(mask_path, image_size):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, [image_size, image_size])
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask

def load_data(image_paths, mask_paths, image_size):
    images = tf.data.Dataset.from_tensor_slices(image_paths)
    masks = tf.data.Dataset.from_tensor_slices(mask_paths)
    images = images.map(lambda x: parse_image(x, image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    masks = masks.map(lambda x: parse_mask(x, image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = tf.data.Dataset.zip((images, masks))
    return dataset

def prepare_dataset(dataset, batch_size):
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResUnet++ model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    args = parser.parse_args()
    file_path = "files/"
    model_path = "files/resunetplusplus_kvasir-SEG_epoch100.h5"
    try:
        os.mkdir("files")
    except:
        pass

    train_path = "data/Kvasir-SEG/train/"
    valid_path = "data/Kvasir-SEG/valid/"

    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = args.epochs

    train_dataset = load_data(train_image_paths, train_mask_paths, image_size)
    valid_dataset = load_data(valid_image_paths, valid_mask_paths, image_size)

    train_dataset = prepare_dataset(train_dataset, batch_size)
    valid_dataset = prepare_dataset(valid_dataset, batch_size)

    nvtx.push_range("Model Building")
    with strategy.scope():
        arch = ResUnetPlusPlus(input_size=image_size)
        model = arch.build_model()
        nvtx.pop_range()

        optimizer = Nadam(lr)
        metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
        model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

    csv_logger = CSVLogger(f"{file_path}unet_{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]

    nvtx.push_range("Model Training")
    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=epochs,
              callbacks=callbacks)
    nvtx.pop_range()

end_time = tf.timestamp()
final_time = end_time - start_time
print(f"Total time taken: {final_time} seconds")

