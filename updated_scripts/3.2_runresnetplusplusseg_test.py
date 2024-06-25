import os
import numpy as np
import tensorflow as tf
import cv2
from glob import glob
from data_generator import DataGen
from tensorflow.keras.models import load_model
from metrics import dice_coef, dice_loss
import time

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
    model_path = "files/resunetplusplus_kvasir-SEG_epoch100.h5"
    test_path = "data/Kvasir-SEG/test/"

    ## Testing
    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "masks", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()

    ## Parameters
    image_size = 256
    batch_size = 8
    test_steps = len(test_image_paths) // batch_size

    ## Generator
    test_gen = DataGen(image_size, test_image_paths, test_mask_paths, batch_size=batch_size)

    ## Load model
    model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})

    ## Evaluate model
    results = model.evaluate(test_gen, steps=test_steps)
    print(f"Test Loss: {results[0]}")
    print(f"Test Metrics: {results[1:]}")

    ## Make predictions
    predictions = model.predict(test_gen, steps=test_steps)

    ## Post-processing predictions and saving example outputs
    output_dir = "files/predictions/"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(predictions)):
        pred_mask = predictions[i]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        image_name = os.path.basename(test_image_paths[i])
        cv2.imwrite(os.path.join(output_dir, image_name), pred_mask)

end_time = tf.timestamp()
final_time = end_time - start_time
print(f"Total time taken: {final_time} seconds")

