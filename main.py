from device import build_strategy
from model import build_model
from dataset import get_dataset

import tensorflow as tf








def main():
    
    strategy, dtype = build_strategy()
    
    with strategy.scope():
        model = build_model()
        optimizer = tf.keras.optimizers.SGD(0.0001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                        reduction=tf.keras.losses.Reduction.NONE)
        
        tracking_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        tracking_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                    'accuracy', dtype=tf.float32)

        # model.compile(optimizer='sgd',
        #                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #                 metrics=['sparse_categorical_accuracy'])
    
    PATH_1 = 'gs://kds-af3ee8c8c3b466fbe7173338006beb4b1548917f9f1b7af696b2a57d'
    PATH_2 = 'gs://kds-545b5300b5e2cdaae853bb7ff6a1fe41a16558a616fd578127034472'
    
    train_shard_suffix = 'train-*-of-01024'
    val_shard_suffix = 'validation-*-of-00128'
    
    train_set_path = tf.io.gfile.glob(PATH_1 + f'/train/{train_shard_suffix}')
    train_set_path += tf.io.gfile.glob(PATH_2 + f'/{train_shard_suffix}')
    val_set_path = tf.io.gfile.glob(PATH_2 + f'/validation/{val_shard_suffix}')
    
    train_set_path = sorted(train_set_path)
    val_set_path = sorted(val_set_path)
    
    relicate = 8
    per_replica_batch_size = 64 
    batch_size = per_replica_batch_size * relicate
    train_set_len = 626000 + 655167# for part 0 and for part 1: 655167
    valid_set_len = 50000
    steps_per_epoch = -(-train_set_len // batch_size)
    validation_steps = -(-valid_set_len // batch_size)

    train_dataset = get_dataset(train_set_path, batch_size, dtype, is_training=True)
    for batch in iter(train_dataset):
        x, y = batch 
        print(x.shape)
        break
    test_dataset = get_dataset(val_set_path, batch_size, dtype, is_training=False)
    for batch in iter(test_dataset):
        x, y = batch 
        print(x.shape)
        break
    
    train_dataset = strategy.distribute_datasets_from_function(
        lambda _: get_dataset(per_replica_batch_size, dtype, is_training=True))
    test_dataset = strategy.distribute_datasets_from_function(
        lambda _: get_dataset(per_replica_batch_size, dtype, is_training=False))
    
    train_iterator = iter(train_dataset)
    test_iterator = iter(test_dataset)

    
    
    @tf.function
    def train_step(iterator):
        """The step function for one training step."""

        def step_fn(inputs):
            """The computation to run on each TPU device."""
            images, labels = inputs
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_fn(labels, logits)

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
            logits = model(images, training=False)
            loss = loss_fn(labels, logits)

            # grads = tape.gradient(loss, model.trainable_variables)
            # optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            tracking_loss.update_state(loss * strategy.num_replicas_in_sync)
            tracking_accuracy.update_state(labels, logits)

        strategy.run(step_fn, args=(next(iterator),))
        
    EPOCH = 200
    for epoch in range(EPOCH):
        print('Epoch: {}/{}'.format(epoch, EPOCH))

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