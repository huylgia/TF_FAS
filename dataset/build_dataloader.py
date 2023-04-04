import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import copy

from . import create_dataset as ds

CLASS_WEIGHTS = None

MEAN = None
STD  = None

ACTIVATION = None
INTERPOLATION = None
TRAIN_BATCH_SIZE = None
VAL_BATCH_SIZE = None
IMAGE_WIDTH = None
IMAGE_HEIGHT = None

PREPROCESS = dict(classes=[0, 1], class_mode='binary' if ACTIVATION=='sigmoid' else 'categorical',
                 target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                 interpolation=INTERPOLATION, shuffle=True, seed=1234)

AUGMENT = dict(rotation_range = 20,
               width_shift_range = 0.2,
               height_shift_range = 0.2,
               shear_range = 0.15,
               zoom_range = 0.15,
               horizontal_flip = True,
               fill_mode="nearest")

def define_classweight(train_df):
    # Define class weight
    class_counts = train_df['class'].value_counts()

    weight0 = len(train_df) / class_counts['live'] * (1 / 2)
    weight1 = len(train_df) / class_counts['spoof'] * (1 / 2)
    class_weight = {0: weight0, 1: weight1}

    return class_weight

def normalize(image):
    mean = np.array(MEAN, dtype=np.float32) * 255.0
    std = np.array(STD, dtype=np.float32) * 255.0

    image = (image.astype(np.float32) - mean) / std
    return image

def build_dataloader():
    train_df, celeba_val_df, lcc_val_df = ds.main()
    CLASS_WEIGHTS = define_classweight(train_df)

    # create datagen
    train_datagen = ImageDataGenerator(preprocessing_function=normalize, **AUGMENT)
    val_datagen   = ImageDataGenerator(preprocessing_function=normalize)

    # get param for dataloader
    train_params = copy.deepcopy(PREPROCESS)
    train_params.update({'dataframe': train_df, 'batch_size': TRAIN_BATCH_SIZE})

    celeba_val_params   = copy.deepcopy(PREPROCESS)
    celeba_val_params.update({'dataframe': celeba_val_df, 'batch_size': VAL_BATCH_SIZE})

    lcc_val_params   = copy.deepcopy(PREPROCESS)
    lcc_val_params.update({'dataframe': lcc_val_df, 'batch_size': VAL_BATCH_SIZE})

    # Generate dataset for train, val and test
    train_gen = train_datagen.flow_from_dataframe(**train_params)
    celeba_val_gen = val_datagen.flow_from_dataframe(**celeba_val_params)
    lcc_val_gen = val_datagen.flow_from_dataframe(**lcc_val_params)

    return train_gen, celeba_val_gen, lcc_val_gen
