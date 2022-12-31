from device import build_strategy
from model import build_model
from dataset import get_dataset
import wandb
import tensorflow as tf
from wandb import WandbMetricsLogger

run = wandb.init(project="test_imagenet_tpu_v2", name="MobilevitSE")


def build_lr_schedule(
        lr_max: float,
        lr_warmup_epochs: int,
        lr_sustain_epochs: int,
        lr_decay: float,
    ):
    def get_lr(epoch: int):
        lr_min = lr_start = lr_max / 100
        if epoch < lr_warmup_epochs:
            lr = (lr_max - lr_start) / lr_warmup_epochs * epoch + lr_start
        elif epoch < lr_warmup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = ((lr_max - lr_min) *
                   lr_decay ** (epoch - lr_warmup_epochs - lr_sustain_epochs) +
                   lr_min)
        return lr
    return get_lr





# def main():
    
strategy, dtype = build_strategy()

PATH_1 = 'gs://kds-af3ee8c8c3b466fbe7173338006beb4b1548917f9f1b7af696b2a57d'
PATH_2 = 'gs://kds-545b5300b5e2cdaae853bb7ff6a1fe41a16558a616fd578127034472'

train_shard_suffix = 'train-*-of-01024'
val_shard_suffix = 'validation-*-of-00128'

train_set_path = tf.io.gfile.glob(PATH_1 + f'/train/{train_shard_suffix}')
train_set_path += tf.io.gfile.glob(PATH_2 + f'/{train_shard_suffix}')
val_set_path = tf.io.gfile.glob(PATH_1 + f'/validation/{val_shard_suffix}')

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
    # print(batch)
    x, y = batch 
    print(x.shape)
    print(tf.math.reduce_min(x), tf.math.reduce_max(x))
    break
test_dataset = get_dataset(val_set_path, batch_size, dtype, is_training=False)
for batch in iter(test_dataset):
    x, y = batch 
    print(x.shape)
    break

lr_schedule = build_lr_schedule(
        lr_max=5e-2,
        lr_warmup_epochs=5,
        lr_sustain_epochs=20,
        lr_decay=0.9,
    )
lr_callback = tf.keras.callbacks.LearningRateScheduler(
    lr_schedule, verbose=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/checkpoints/model.hdf5',
                            #  monitor='loss',
                             save_freq='epoch',
                             verbose=1,
                             period=3,
                             verbose=True,
                             # save_best_only=True,
                             save_weights_only=True,
                             )
wandb_callback = WandbMetricsLogger()
from mobilevit import create_mobilevit
with strategy.scope():
    model = create_mobilevit()
    optimizer = tf.keras.optimizers.SGD()
    # loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE)
    
    tracking_loss = tf.keras.metrics.Mean('loss')
    tracking_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                'accuracy')
    


    


    # model.compile(optimizer='sgd',
    #                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                 metrics=['sparse_categorical_accuracy'])

model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=200,
        # callbacks=callbacks,
        validation_freq=3,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=[lr_callback, cp_callback, wandb_callback]
    )


assert False

train_dataset = strategy.distribute_datasets_from_function(
    lambda _: get_dataset(train_set_path, per_replica_batch_size, dtype, is_training=True))
test_dataset = strategy.distribute_datasets_from_function(
    lambda _: get_dataset(val_set_path, per_replica_batch_size, dtype, is_training=False))

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
        # loss = tf.nn.compute_average_loss(loss, global_batch_size=batch_size)

        # grads = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
        tracking_loss.update_state(loss * strategy.num_replicas_in_sync)
        tracking_accuracy.update_state(labels, logits)

    strategy.run(step_fn, args=(next(iterator),))

EPOCH = 200
import time 
time1 = time.time()

# for step in range(5): #steps_per_epoch
#     train_step(train_iterator)
# print("Loss:",tracking_loss.result())
# tracking_loss.reset_states()

for epoch in range(EPOCH):
    print('Epoch: {}/{}'.format(epoch, EPOCH))

    # for _ in range(100)
    for step in range(1,steps_per_epoch+1): #steps_per_epoch
        train_step(train_iterator)
        if step % 100 == 0:
            
            print('Current step: {}, training loss: {}, accuracy: {}%'.format(
                optimizer.iterations.numpy(),
                round(float(tracking_loss.result()), 4),
                round(float(tracking_accuracy.result()) * 100, 4)))
            tracking_loss.reset_states()
            tracking_accuracy.reset_states()
    # tracking_loss.reset_states()
    # tracking_accuracy.reset_states()
    # print("Loss:",tracking_loss.result())
    # tracking_loss.reset_states()
    
    for step in range(validation_steps): #validation_steps
        test_step(test_iterator)
    print('Current step: {}, validation loss: {}, accuracy: {}%'.format(
        epoch,
        round(float(tracking_loss.result()), 4),
        round(float(tracking_accuracy.result()) * 100, 4)))
    tracking_loss.reset_states()
    tracking_accuracy.reset_states()

    time_running = time.time()-time1
    print("time_running:", time_running)
    time1 = time.time()
        
# if __name__ == "__main__":
#     main()