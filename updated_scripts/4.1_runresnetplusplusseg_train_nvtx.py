import os
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from data_generator import DataGen
from m_resunet import ResUnetPlusPlus
from metrics import dice_coef, dice_loss
import argparse
import time
import nvtx  # Import NVTX

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

start_time = tf.timestamp()

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

    ## Training
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    ## Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = args.epochs 
    train_steps = len(train_image_paths) // batch_size
    valid_steps = len(valid_image_paths) // batch_size

    ## Generator
    nvtx.push_range("DataGen initialization")
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)
    nvtx.pop_range()

    ## ResUnet++
    nvtx.push_range("Model Building")
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
    model.fit(train_gen,
              validation_data=valid_gen,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              epochs=epochs,
              callbacks=callbacks)
    nvtx.pop_range()

end_time = tf.timestamp()
final_time = end_time - start_time
print(f"Total time taken: {final_time} seconds")

