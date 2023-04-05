import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten


input_idx = 0
dataset = tf.data.Dataset.load(f'taus_{input_idx}', compression='GZIP')

class TauL1Model(keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv = [
      Conv2D(8, (1, 1), activation='relu'),
      Conv2D(8, (1, 1), activation='relu'),
      Conv2D(8, (3, 3), activation='relu'),
      Conv2D(8, (3, 3), activation='relu'),
      Conv2D(8, (2, 2), activation='relu'),
    ]
    self.flatten = Flatten()
    self.dense = [
      Dense(32, activation='relu'),
      Dense(16, activation='relu'),
      Dense(1, activation='sigmoid'),
    ]

  @tf.function
  def call(self, x):
    for conv in self.conv:
      x = conv(x)
    x = self.flatten(x)
    for dense in self.dense:
      x = dense(x)
    return x

model = TauL1Model()
model.compile(optimizer='adam', loss='binary_crossentropy', weighted_metrics=['accuracy'])


def to_train(x, y, w, meta):
  return x[:4], y, w

ds_train_val = dataset.batch(300).map(to_train)
n_batches = ds_train_val.cardinality().numpy()
n_batches_train = int(n_batches * 0.8)
ds_train = ds_train_val.take(n_batches_train)
ds_val = ds_train_val.skip(n_batches_train)

for x, y, w in ds_train:
  model(x)
  break
model.summary()

model.fit(ds_train, validation_data=ds_val, epochs=10, verbose=1)
model.save('models/model_v0')