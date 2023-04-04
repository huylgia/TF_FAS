from keras.callbacks import Callback

class CustomLogger(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.model.optimizer._learning_rate(self.model.optimizer.iterations)
        print(f" - lr: {lr:.6f}")

    def on_train_batch_end(self, batch, logs=None):
      write_content(self.log_file, logs, batch)

    def on_test_batch_end(self, batch, logs=None):
      write_content(self.log_file, logs, batch)

    def on_predict_batch_end(self, batch, logs=None):
      write_content(self.log_file, logs, batch)

def write_content(log_file, logs, batch):
    with open(log_file, 'a') as f:
        form_string = "{}: {:>1.4f}"
        content = []
        for key, value in logs.items():
          content.append(form_string.format(key, value))
        
        content = ", ".join(content)
        f.write(f"Step {batch}: {content}\n")