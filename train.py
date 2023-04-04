from keras_cv_attention_models import edgenext, fasternet
from loss import macro_f1_loss, binary_focal_crossentropy, binary_crossentropy, combine_loss

ACTIVATION = 'sigmoid'
INTERPOLATION = 'bilinear'
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

config = dict(
    MEAN = [0.485, 0.456, 0.406],
    STD  = [0.229, 0.224, 0.225],

    ACTIVATION = ACTIVATION,
    TRAIN_BATCH_SIZE = 1024,
    VAL_BATCH_SIZE = 1024,
    CLASS_WEIGHTS = False,

    PREPROCESS = dict(classes=['0', '1'], class_mode='binary' if ACTIVATION=='sigmoid' else 'categorical',
                    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                    interpolation=INTERPOLATION, shuffle=True, seed=1234),

    AUGMENT = dict(rotation_range = 20,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.15,
                zoom_range = 0.15,
                horizontal_flip = True,
                fill_mode="nearest"),
                        
    MODEL_NAME   = edgenext.EdgeNeXt_XX_Small,
    MODEL_PARAMS = dict(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                        pretrained="imagenet", num_classes=1, classifier_activation=ACTIVATION,
                        dropout=0.5),
    MODEL_DIR= "/content/drive/MyDrive/Liveness/outputs/tensorflow/edgenext_xx_small",
    SAVE_EPOCH=2,
    EPOCH=10,

    LOSS = binary_focal_crossentropy)