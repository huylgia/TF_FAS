import dataset
from metric import ACER
from callback import CustomLogger, CSVLogger, ModelCheckpoint
from optimizer import WarmUp
from keras.models import load_model
import os

def setup_callback(total_step, config):
    if not os.path.exists(config['MODEL_DIR']):
        os.makedirs(config['MODEL_DIR'])

    # train logger
    train_logger = CustomLogger(config['MODEL_DIR']+'/training.log')

    # checkpoint
    checkpoint_call    = ModelCheckpoint(config['MODEL_DIR'] + '/best_acer.h5', mode='min', monitor='val_acer', save_best_only=True, verbose=1)

    # save after n epoch
    epoch_path = config['MODEL_DIR'] + '/epoch_{epoch:02d}.h5'
    epoch_call = ModelCheckpoint(epoch_path, save_freq=config['SAVE_EPOCH']*total_step, save_best_only=True, verbose=1)

    # save latest
    latest_path = config['MODEL_DIR'] + "/latest.h5"
    latest_call = ModelCheckpoint(latest_path, verbose=1)

    return [checkpoint_call, train_logger, val_logger, epoch_call, latest_call]

def trainer(config):
    # build dataloader
    train_gen, val_gen = dataset.build_dataloader(config)
    total_step = len(train_gen)

    # set up scheduler
    config['SCHEDULER_PARAMS']['decay_steps'] = total_step//5
    
    warmup_epoch = config['SCHEDULER_PARAMS'].pop('warmup_epoch', -1)
    warmup_steps = config['SCHEDULER_PARAMS'].pop('warmup_steps', 0)
    scheduler = WarmUp(**config['SCHEDULER_PARAMS'], decay_schedule_fn=config['SCHEDULER'](**config['SCHEDULER_PARAMS']),
                       step_per_epoch=total_step, 
                       warmup_epoch=warmup_epoch,
                       warmup_steps=warmup_steps)

    # set up optimizer
    config['OPTIMIZER_PARAMS']['learning_rate'] = scheduler
    optimizer = config['OPTIMIZER'](**config['OPTIMIZER_PARAMS'])

    # set up metric
    metrics = ['binary_accuracy' if config['ACTIVATION']=='sigmoid' else 'accuracy', ACER(name="apcer"), ACER(name="bpcer"), ACER(name="acer")]

    # set up loss
    loss = config['LOSS']

    # set up callback
    callbacks = setup_callback(total_step, config)

    # compile model
    model = config['MODEL'](**config['MODEL_PARAMS'])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    if config['RESUME']:
        config['RESUME_PARAMS'].update({'ACER': ACER, 'WarmUp': WarmUp})
        model = load_model(config['RESUME'], custom_objects=config['RESUME_PARAMS'])
        
    # call fit
    model.fit(train_gen, validation_data=val_gen, epochs=config['EPOCH'],
              steps_per_epoch=total_step,
              callbacks=callbacks,
              class_weight=config['CLASS_WEIGHTS'])






