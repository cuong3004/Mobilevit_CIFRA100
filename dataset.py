import tensorflow_datasets as tfds
import tensorflow as tf
from transform import RandomAugmentor

def load_imagenet(tfrec_paths):
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(tfrec_paths, num_parallel_reads=AUTO)
    
    options_no_order = tf.data.Options()
    options_no_order.experimental_deterministic = False
    dataset = dataset.with_options(options_no_order)
    
    def deserialization_fn(serialized_example):
        parsed_example = tf.io.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            }
        )
        image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)
        # image = tf.image.resize(image, image_shape)
        label = tf.cast(parsed_example['image/class/label'], tf.int64) - 1  # [0-999]
        return image, label

    dataset.map(deserialization_fn)
    
    return dataset

def load_cifar(data_dir, is_training=True):
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='cifar100', data_dir=data_dir, split=split, with_info=True,
                                as_supervised=True, try_gcs=True)

def get_dataset(tfrec_paths, batch_size, dtype, image_size=(224,224), is_training=True):
    

    
    # load_cifar(data_dir, is_training)
    


    # Normalize the input data.
    dataset = load_imagenet(tfrec_paths)

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