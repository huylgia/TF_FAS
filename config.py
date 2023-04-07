from keras_cv_attention_models import edgenext, efficientformer
from loss import macro_f1_loss, binary_focal_crossentropy, binary_crossentropy, combine_loss
from optimizer import ExponentialDecay, Lion, Adam

ACTIVATION = 'sigmoid'
INTERPOLATION = 'bilinear'
IMAGE_WIDTH = 112
IMAGE_HEIGHT = 112

config = dict(
    TRAIN_DIS=True,
    SAVE_EPOCH=2,
    EPOCH=10,

    # =====================DATASET======================
    CELEBA_TRAIN_DIRECTORY = "../../dataset/Liveness/CelebA_Spoof_origin_crop/Data/train",
    CELEBA_VAL_DIRECTORY  = "../../dataset/Liveness/CelebA_Spoof_origin_crop/Data/test",
    LCC_TRAIN_DIRECTORY = "../../dataset/Liveness/LCC_FASD/LCC_FASD_training",
    LCC_VAL_DIRECTORY = "../../dataset/Liveness/LCC_FASD/LCC_FASD_development",

    MEAN = [0.485, 0.456, 0.406],
    STD  = [0.229, 0.224, 0.225],

    ACTIVATION = ACTIVATION,
    TRAIN_BATCH_SIZE = 256,
    VAL_BATCH_SIZE = 256,
    CLASS_WEIGHTS = None,

    PREPROCESS = dict(classes=['0', '1'], class_mode='binary' if ACTIVATION=='sigmoid' else 'categorical',
                    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                    interpolation=INTERPOLATION, shuffle=True, seed=1234),

    # AUGMENT = dict(rotation_range = 20,
    #             width_shift_range = 0.2,
    #             height_shift_range = 0.2,
    #             shear_range = 0.15,
    #             zoom_range = 0.15,
    #             horizontal_flip = True,
    #             fill_mode="nearest"),

    # ======================MODEL============================         
    MODEL = efficientformer.EfficientFormerV2S0,
    MODEL_PARAMS = dict(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                        pretrained="imagenet", num_classes=1, classifier_activation=ACTIVATION,
                        dropout=0.5, use_distillation=False),
                        
    MODEL_DIR = "outputs/eformerv2_070423",
    # RESUME = "/content/drive/MyDrive/Liveness/outputs/tensorflow/eformerv2/latest.h5",
    # RESUME_PARAMS = dict(Lion = Lion),

    # =====================OPTIMIZER========================
    SCHEDULER_PARAMS = dict(initial_learning_rate=0.001, decay_rate=0.96, warmup_epoch=2),
    SCHEDULER = ExponentialDecay,

    OPTIMIZER_PARAMS = dict(weight_decay=0.1, beta_1=0.9, beta_2=0.99),
    OPTIMIZER = Lion,

    # =====================LOSS==============================
    LOSS = combine_loss,
    LOSS_PARAMS=dict(alpha_1=0.75, alpha_2=0.25)
    )