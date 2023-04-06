from keras.callbacks import Callback

class CustomLogger(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log = open(log_file, 'a')

    def on_epoch_begin(self, epoch, logs=None):
        with open(self.log_file, 'a') as f:
            f.write('Starting epoch {}'.format(epoch+1))

    def on_train_batch_end(self, batch, logs=None):
        lr = self.model.optimizer._learning_rate(self.model.optimizer.iterations)
        logs['lr'] = lr

        content = create_content(logs)
        f.write('\tStep {}: {}'.format(batch, content))

    def on_test_end(self, logs=None):
        content = write_content(logs)
        f.write("\tEvaluate: {}".format(content))

def create_content(logs):
    form_string = "{}: {:>1.4f}"
    content = []
    for key, value in logs.items():
        content.append(form_string.format(key, value))
    
    content = ", ".join(content)

    return content