import tensorflow_datasets as tfds
import tensorflow as tf

def get_dataset(batch_size, is_training=True):
  split = 'train' if is_training else 'test'
  dataset, info = tfds.load(name='mnist', split=split, with_info=True,
                            as_supervised=True, try_gcs=True)

  # Normalize the input data.
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

  dataset = dataset.map(scale)

  # Only shuffle and repeat the dataset in training. The advantage of having an
  # infinite dataset for training is to avoid the potential last partial batch
  # in each epoch, so that you don't need to think about scaling the gradients
  # based on the actual batch size.
  if is_training:
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()

  dataset = dataset.batch(batch_size)

  return dataset