import os
import pandas as pd
import numpy as np

LABEL_MAP = {'real': '0', 'live': '0', 'fake': '1', 'spoof': '1'}

def main(config):
    celeba_train_df = create_celeba_df(config['CELEBA_TRAIN_DIRECTORY'])
    celeba_val_df = create_celeba_df(config['CELEBA_VAL_DIRECTORY'])
    lcc_fasd_train_df = create_lcc_fasd_df(config['LCC_TRAIN_DIRECTORY'])
    lcc_fasd_val_df = create_lcc_fasd_df(config['LCC_VAL_DIRECTORY'])

    # concat training dataset
    train_df = pd.concat([celeba_train_df, lcc_fasd_train_df])
    val_df = pd.concat([celeba_val_df, lcc_fasd_val_df])

    return train_df, val_df

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
        if os.path.isfile(label_dir):
            continue

        data = [["{}/{}".format(label_dir, filename), LABEL_MAP[label]] for filename in os.listdir(label_dir)]
        df.extend(data)

    return pd.DataFrame(df, columns=['filename', 'class'])
