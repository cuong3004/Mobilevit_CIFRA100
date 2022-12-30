import tensorflow as tf

def build_model(num_classes=100,):
    backbone = tf.keras.applications.ResNet101V2(
        input_shape=(224,224,3),
        weights="imagenet",
        # weights=None,
    )

    x = backbone.layers[-2].output
    x = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(backbone.input, x)


    
    # return model
    # return tf.keras.Sequential(
    #   [tf.keras.layers.Conv2D(256, 3, activation='relu', input_shape=(32, 32, 3)),
    #    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(256, activation='relu'),
    #    tf.keras.layers.Dense(128, activation='relu'),
    #    tf.keras.layers.Dense(10)])