from device import build_strategy
from model import build_model
from dataset import get_dataset

import tensorflow as tf

def main():
    
    strategy = build_strategy()
    
    with strategy.scope():
        model = build_model()
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['sparse_categorical_accuracy'])
        
        batch_size = 200
        steps_per_epoch = 60000 // batch_size
        validation_steps = 10000 // batch_size

        train_dataset = get_dataset(batch_size, is_training=True)
        test_dataset = get_dataset(batch_size, is_training=False)
        
        model.fit(train_dataset,
            epochs=5,
            steps_per_epoch=steps_per_epoch,
            validation_data=test_dataset,
            validation_steps=validation_steps)


if __name__ == "__main__":
    main()