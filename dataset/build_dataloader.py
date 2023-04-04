import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import copy

from . import create_dataset as ds

MEAN = None
STD  = None

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

def build_dataloader(config):
    global MEAN, STD
    MEAN = config['MEAN']
    STD  = config['STD']
    
    train_df, celeba_val_df, lcc_val_df = ds.main()
    config['CLASS_WEIGHTS'] = define_classweight(train_df) if config['CLASS_WEIGHTS'] else None

    # create datagen
    train_datagen = ImageDataGenerator(preprocessing_function=normalize, **config['AUGMENT'])
    val_datagen   = ImageDataGenerator(preprocessing_function=normalize)

    # get param for dataloader
    train_params = copy.deepcopy(config['PREPROCESS'])
    train_params.update({'dataframe': train_df, 'batch_size': config['TRAIN_BATCH_SIZE']})

    celeba_val_params   = copy.deepcopy(config['PREPROCESS'])
    celeba_val_params.update({'dataframe': celeba_val_df, 'batch_size': config['VAL_BATCH_SIZE']})

    lcc_val_params   = copy.deepcopy(config['PREPROCESS'])
    lcc_val_params.update({'dataframe': lcc_val_df, 'batch_size': config['VAL_BATCH_SIZE']})

    # Generate dataset for train, val and test
    train_gen = train_datagen.flow_from_dataframe(**train_params)
    celeba_val_gen = val_datagen.flow_from_dataframe(**celeba_val_params)
    lcc_val_gen = val_datagen.flow_from_dataframe(**lcc_val_params)

    return train_gen, celeba_val_gen, lcc_val_gen
