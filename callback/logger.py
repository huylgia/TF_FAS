from keras.callbacks import Callback

class CustomLogger(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log = open(log_file, 'a')

    def on_epoch_begin(self, epoch, logs=None):
        self.log.write('Starting epoch {}\n'.format(epoch+1))

    def on_train_batch_end(self, batch, logs=None):
        lr = self.model.optimizer._learning_rate(self.model.optimizer.iterations)
        logs['lr'] = lr

        content = create_content(logs)
        self.log.write('\tStep {}: {}\n'.format(batch, content))

    def on_test_end(self, logs=None):
        content = create_content(logs)
        self.log.write("\tEvaluate: {}\n".format(content))

def create_content(logs):
    form_string = "{}: {:>1.4f}"
    content = []
    for key, value in logs.items():
        content.append(form_string.format(key, value))
    
    content = ", ".join(content)

    return content
