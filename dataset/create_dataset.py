import os
import pandas as pd
import numpy as np

LABEL_MAP = {'real': 0, 'live': 0, 'fake': 1, 'spoof': 1}

CELEBA_TRAIN_DIRECTORY = "/content/CelebA_Spoof_origin_crop/Data/train"
CELEBA_VAL_DIRECTORY  = "/content/CelebA_Spoof_origin_crop/Data/test"
LCC_TRAIN_DIRECTORY = "/content/LCC_FASD/LCC_FASD_training"
LCC_VAL_DIRECTORY = "/content/LCC_FASD/LCC_FASD_development"

def main():
    celeba_train_df = create_celeba_df(CELEBA_TRAIN_DIRECTORY)
    celeba_val_df = create_celeba_df(CELEBA_VAL_DIRECTORY)
    lcc_fasd_train_df = create_lcc_fasd_df(LCC_TRAIN_DIRECTORY)
    lcc_fasd_val_df = create_lcc_fasd_df(LCC_VAL_DIRECTORY)

    # concat training dataset
    train_df = pd.concat([celeba_train_df, lcc_fasd_train_df])

    return train_df, celeba_val_df, lcc_fasd_val_df

def create_celeba_df(directory):
    '''
        labels: live, spoof
    '''
    df = []

    face_ids = os.listdir(directory)
    for face_id in face_ids:
        face_id_dir = os.path.join(directory, face_id)
        for label in os.listdir(face_id_dir):
            label_dir = os.path.join(face_id_dir, label)

            data = [["{}/{}/{}".format(face_id_dir, label, filename), LABEL_MAP[label]] for filename in os.listdir(label_dir)]
            df.extend(data)

    return pd.DataFrame(df, columns=['filename', 'class'])

def create_lcc_fasd_df(directory):
    '''
        labels: real, fake
    '''
    df = []

    for label in os.listdir(directory):
        if label.startswith("."): continue
        
        label_dir = os.path.join(directory, label)

        data = [["{}/{}".format(label_dir, filename), LABEL_MAP[label]] for filename in os.listdir(label_dir)]
        df.extend(data)

    return pd.DataFrame(df, columns=['filename', 'class'])
