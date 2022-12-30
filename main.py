from device import build_strategy
from model import build_model
from dataset import get_dataset

import tensorflow as tf








def main():
    
    strategy, dtype = build_strategy()
    
    with strategy.scope():
        model = build_model()
        optimizer = tf.keras.optimizers.SGD()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        tracking_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        tracking_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    'accuracy', dtype=tf.float32)

        # model.compile(optimizer='sgd',
        #                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #                 metrics=['sparse_categorical_accuracy'])
        
    relicate = 8
    per_replica_batch_size = 64 
    batch_size = per_replica_batch_size * relicate
    steps_per_epoch = 50000 // batch_size
    validation_steps = 10000 // batch_size

    train_dataset = get_dataset(batch_size, dtype, is_training=True)
    test_dataset = get_dataset(batch_size, dtype, is_training=False)
    
    train_dataset = strategy.experimental_distribute_datasets_from_function(
        lambda _: get_dataset(per_replica_batch_size, is_training=True))
    test_dataset = strategy.experimental_distribute_datasets_from_function(
        lambda _: get_dataset(per_replica_batch_size, is_training=False))
    
    train_iterator = iter(train_dataset)
    test_iterator = iter(test_dataset)

    for batch in iter(train_dataset):
        x, y = batch 
        print(x.shape)
        break
    
    @tf.function
    def train_step(iterator):
        """The step function for one training step."""

        def step_fn(inputs):
            """The computation to run on each TPU device."""
            images, labels = inputs
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_fn(labels, logits, from_logits=True)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
                tracking_loss.update_state(loss * strategy.num_replicas_in_sync)
                tracking_accuracy.update_state(labels, logits)

        strategy.run(step_fn, args=(next(iterator),))
        
    @tf.function
    def test_step(iterator):
        """The step function for one training step."""

        def step_fn(inputs):
            """The computation to run on each TPU device."""
            images, labels = inputs
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_fn(labels, logits, from_logits=True)

                # grads = tape.gradient(loss, model.trainable_variables)
                # optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
                tracking_loss.update_state(loss * strategy.num_replicas_in_sync)
                tracking_accuracy.update_state(labels, logits)

        strategy.run(step_fn, args=(next(iterator),))
        
    EPOCH = 200
    for epoch in range(EPOCH):
        print('Epoch: {}/{EPOCH}'.format(epoch))

        for step in range(steps_per_epoch):
            train_step(train_iterator)
        print('Current step: {}, training loss: {}, accuracy: {}%'.format(
            optimizer.iterations.numpy(),
            round(float(tracking_loss.result()), 4),
            round(float(tracking_accuracy.result()) * 100, 4)))
        tracking_loss.reset_states()
        tracking_accuracy.reset_states()
        
        for step in range(validation_steps):
            test_step(test_iterator)
        print('Current step: {}, validation loss: {}, accuracy: {}%'.format(
            optimizer.iterations.numpy(),
            round(float(tracking_loss.result()), 4),
            round(float(tracking_accuracy.result()) * 100, 4)))
        tracking_loss.reset_states()
        tracking_accuracy.reset_states()
        

if __name__ == "__main__":
    main()