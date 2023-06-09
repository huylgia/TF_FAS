from typing import Callable
import tensorflow as tf

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(
      self,
      initial_learning_rate: float,
      decay_schedule_fn: Callable,
      warmup_steps: int = 0,
      power: float = 1.0,
      name: str = None,
      warmup_epoch: int = -1,
      step_per_epoch: int = -1,
      **kwargs
  ):
      super().__init__()
      if warmup_epoch >= 0:
        assert step_per_epoch >= 0, "When call warmup_epoch, need step_per_epoch"

      self.initial_learning_rate = initial_learning_rate
      self.warmup_steps = step_per_epoch*warmup_epoch if warmup_epoch >= 0 else warmup_steps
      self.power = power
      self.decay_schedule_fn = decay_schedule_fn
      self.name = name

  def __call__(self, step):
      with tf.name_scope(self.name or "WarmUp") as name:
          # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
          # learning rate will be `global_step/num_warmup_steps * init_lr`.
          global_step_float = tf.cast(step, tf.float32)
          warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
          warmup_percent_done = global_step_float / warmup_steps_float
          warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
          return tf.cond(
              global_step_float < warmup_steps_float,
              lambda: warmup_learning_rate,
              lambda: self.decay_schedule_fn(step - self.warmup_steps),
              name=name,
          )

  def get_config(self):
      return {
          "initial_learning_rate": self.initial_learning_rate,
          "decay_schedule_fn": self.decay_schedule_fn,
          "warmup_steps": self.warmup_steps,
          "power": self.power,
          "name": self.name,
      }