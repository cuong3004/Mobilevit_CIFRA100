import tensorflow_datasets as tfds
import tensorflow as tf
from transform import RandomAugmentor

def get_dataset(batch_size, dtype, image_size=(224,224), is_training=True):
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='cifar100', data_dir="gs://cuong_tpu/data", split=split, with_info=True,
                                as_supervised=True, try_gcs=True)

    # Normalize the input data.

    bt_augmentor = RandomAugmentor(image_size[0])

    def process(image, label):
        # image = tf.image.resize_and_crop_image(
        #     image, target_size=image_size)
        image = tf.image.resize(image, (image_size))
        image = bt_augmentor(image)

        

        image = tf.cast(image, dtype)
        image /= 255.0
        return image, label

    def process_valid(image, label):
        image = tf.image.resize(image, (image_size))
        image = tf.cast(image, dtype)
        image /= 255.0
        return image, label

#   dataset = dataset.map(scale)
    AUTO = tf.data.experimental.AUTOTUNE

    if is_training:
        print("Is train")
        dataset = dataset.map(process, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(process_valid, num_parallel_calls=AUTO)

    dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(4096)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset