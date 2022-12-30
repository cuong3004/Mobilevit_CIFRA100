import tensorflow as tf

def build_model(num_classes=1000, ):
    # model = tf.keras.applications.MobileNetV2(
    #     input_shape=(224,224,3),
    #     include_top=True,
    #     weights="imagenet",
    #     input_tensor=None,
    #     pooling=None,
    #     classes=num_classes,
    #     classifier_activation="softmax",
    # )
    
    # return model
    return tf.keras.Sequential(
      [tf.keras.layers.Conv2D(256, 3, activation='relu', input_shape=(28, 28, 1)),
       tf.keras.layers.Conv2D(256, 3, activation='relu'),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(256, activation='relu'),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)])