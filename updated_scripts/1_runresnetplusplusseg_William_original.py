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
####################################
#added aditionall pacages for GPU and
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19
####################################
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
device = torch.device("cuda:2")
####################################
start_time = time.time()
if __name__ == "__main__":
    ## Path
    # paths renamed for proper files
    file_path = "files/"
    model_path = "files/resunetplusplus_kvasir-SEG_epoch100.h5" #renamed for clarity
    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass
    #paths renamed to proper loactions
    train_path = "new_data/Kvasir-SEG27/train/"
    valid_path = "new_data/Kvasir-SEG27/valid/"
    ## Training
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()
    # train_image_paths = train_image_paths[:2000]
    # train_mask_paths = train_mask_paths[:2000]
    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()
    ## Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 100 #200 #Number of epochs changed
    train_steps = len(train_image_paths)//batch_size
    valid_steps = len(valid_image_paths)//batch_size
    ## Generator
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)
    ## Unet
    #arch = Unet(input_size=image_size)
    #model = arch.build_model()
    ## ResUnet
    #arch = ResUnet(input_size=image_size)
    #model = arch.build_model()
    ## ResUnet++
    arch = ResUnetPlusPlus(input_size=image_size)
    model = arch.build_model()
    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
    model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)
    csv_logger = CSVLogger(f"{file_path}unet_{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]
    model.fit(train_gen,
            validation_data=valid_gen,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            epochs=epochs,
            callbacks=callbacks)
end_time = time.time()
final_time = end_time - start_time
