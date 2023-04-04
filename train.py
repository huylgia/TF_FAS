import program
from program import dataset
from keras_cv_attention_models import edgenext, fasternet
from loss import macto_f1_loss, binary_focal_crossentropy, binary_crossentropy, combine_loss

dataset.MEAN = [0.485, 0.456, 0.406]
dataset.STD  = [0.229, 0.224, 0.225]

dataset.ACTIVATION = 'sigmoid'
dataset.INTERPOLATION = 'bilinear'
dataset.TRAIN_BATCH_SIZE = 1024
dataset.VAL_BATCH_SIZE = 1024
dataset.IMAGE_WIDTH = 128
dataset.IMAGE_HEIGHT = 128
dataset.CLASS_WEIGHTS = False

TRAIN_PARAMS = dict(input_shape=(dataset.IMAGE_HEIGHT, dataset.IMAGE_WIDTH, 3),
                    pretrained="imagenet", num_classes=1, classifier_activation=dataset.ACTIVATION,
                    dropout=0.5)
                    
program.MODEL= edgenext.EdgeNeXt_XX_Small(**TRAIN_PARAMS)
program.MODEL_DIR= "/content/drive/MyDrive/Liveness/outputs/tensorflow/edgenext_xx_small"
program.SAVE_EPOCH=2
program.EPOCH=10

program.LOSS = binary_focal_crossentropy


