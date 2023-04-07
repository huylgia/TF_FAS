from metric import ACER
from keras.models import load_model
import os
import tensorflow as tf

CKPT_FILE = 

def evaluator(config):
    # set up metric
    metrics = [ACER(name="apcer"), ACER(name="bpcer"), ACER(name="acer")]

    # 
    model = load_model(CKPT_FILE, compile=False)
    model.compile(metrics=metrics)







