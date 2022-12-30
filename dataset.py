import tensorflow_datasets as tfds
import tensorflow as tf
from transform import RandomAugmentor

def load_imagenet(tfrec_paths):
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(tfrec_paths, num_parallel_reads=AUTO)
    
    options_no_order = tf.data.Options()
    options_no_order.experimental_deterministic = False
    dataset = dataset.with_options(options_no_order)
    
    return dataset

def load_cifar(data_dir, is_training=True):
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='cifar100', data_dir=data_dir, split=split, with_info=True,
                                as_supervised=True, try_gcs=True)

def get_dataset(tfrec_paths, batch_size, dtype, image_size=(224,224), is_training=True):
    
    bt_augmentor = RandomAugmentor(image_size[0])

    def process(image):
        image = tf.image.resize(image, (image_size))
        image = bt_augmentor(image)

        image = tf.cast(image, dtype)
        image /= 255.0
        return image

    def process_valid(image):
        image = tf.image.resize(image, (image_size))
        image = tf.cast(image, dtype)
        image /= 255.0
        return image

    
    def decode(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            })
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.image.resize(image, (image_size))
        image = bt_augmentor(image) if is_training else image
        image = tf.cast(image, dtype)
        image /= 255.0
        label = tf.cast(features['image/class/label'], tf.int64) - 1  # [0-999]
        return image, label

    def create_ds(tfrecords, batch_size):
        dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(4096) if is_training else dataset
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    dataset = create_ds(tfrec_paths, batch_size)

    
    
    

    
    

    return dataset