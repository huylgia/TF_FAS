import dataset
from metric import ACER
from callback import CustomLogger, CSVLogger, ModelCheckpoint
from optimizer import WarmUp, ExponentialDecay, Lion, Adam
import os

MODEL = None
MODEL_DIR = None
LOSS = None
SAVE_EPOCH = None
EPOCH = None

def setup_callback(total_step):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # train logger
    train_logger = CustomLogger(MODEL_DIR+'/training.log')

    # val logger 
    val_logger = CSVLogger(MODEL_DIR+'/evaluate.log')

    # checkpoint
    celeba_call = ModelCheckpoint(MODEL_DIR + '/best_celeba.h5', mode='min', monitor='val_acer', save_best_only=True, verbose=1)
    lcc_call    = ModelCheckpoint(MODEL_DIR + '/best_lcc_fasd.h5', mode='min', monitor='val_acer', save_best_only=True, verbose=1)

    # save after n epoch
    epoch_path = MODEL_DIR + '/epoch_{epoch:02d}.h5'
    epoch_call = ModelCheckpoint(epoch_path, save_freq=SAVE_EPOCH*total_step, save_best_only=True, verbose=1)

    # save latest
    latest_path = MODEL_DIR + "/latest.h5"
    latest_call = ModelCheckpoint(latest_path, verbose=1)

    return [celeba_call, lcc_call, train_logger, val_logger, epoch_call, latest_call]

def trainer():
    # build dataloader
    train_gen, celeba_val_gen, lcc_val_gen = dataset.build_dataloader()
    total_step = len(train_gen)

    # set up scheduler
    scheduler_params = dict(initial_learning_rate=0.001, decay_rate=0.96,
                            decay_steps=total_step//5)
    scheduler = WarmUp(**scheduler_params, decay_schedule_fn=ExponentialDecay(**scheduler_params),
                    step_per_epoch=total_step, warmup_epoch=1)

    # set up optimizer
    optimizer = Lion(learning_rate=scheduler, weight_decay=0.1, beta_1=0.9, beta_2=0.99)

    # set up metric
    metrics = ['binary_accuracy' if dataset.ACTIVATION=='sigmoid' else 'accuracy', ACER(name="apcer"), ACER(name="bpcer"), ACER(name="acer")]

    # set up loss
    loss = LOSS

    # set up callback
    callbacks = setup_callback(total_step)

    # compile model
    MODEL.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # call fit
    MODEL.fit(train_gen, validation_data=[celeba_val_gen, lcc_val_gen], epochs=EPOCH, steps_per_epoch=total_step)






