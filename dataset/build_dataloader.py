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

    weight0 = len(train_df) / class_counts['0'] * (1 / 2)
    weight1 = len(train_df) / class_counts['1'] * (1 / 2)
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
    
    train_df, val_df = ds.main(config)
    config['CLASS_WEIGHTS'] = define_classweight(train_df) if config['CLASS_WEIGHTS'] else None

    # create datagen
    train_datagen = ImageDataGenerator(preprocessing_function=normalize, **config.get('AUGMENT', dict()))
    val_datagen   = ImageDataGenerator(preprocessing_function=normalize)

    # get param for dataloader
    train_params = copy.deepcopy(config['PREPROCESS'])
    train_params.update({'dataframe': train_df, 'batch_size': config['TRAIN_BATCH_SIZE']})

    val_params   = copy.deepcopy(config['PREPROCESS'])
    val_params.update({'dataframe': val_df, 'batch_size': config['VAL_BATCH_SIZE']})


    # Generate dataset for train, val and test
    train_gen = train_datagen.flow_from_dataframe(**train_params)
    val_gen = val_datagen.flow_from_dataframe(**val_params)

    return train_gen, val_gen
