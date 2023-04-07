from keras.losses import binary_focal_crossentropy, binary_crossentropy
from .macro_f1_loss import macro_f1_loss

def combine_loss(y_true, y_pred):
    loss_1 = macro_f1_loss(y_true, y_pred, alpha=0.75)
    loss_2 = binary_focal_crossentropy(y_true, y_pred, alpha=0.25)

    return loss_1 + loss_2