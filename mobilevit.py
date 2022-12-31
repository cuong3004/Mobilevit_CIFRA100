

from typing import Tuple
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from tensorflow.keras.layers import EinsumDense
except:
    from tensorflow.keras.layers.experimental import EinsumDense


# Values are from table 4.
patch_size = 4  # 2x2, for the Transformer blocks.
image_size = 256
expansion_factor = 4  # expansion factor for the MobileNetV2 blocks.

import math
class MyMultiHeadAttention(layers.Layer):
    def __init__(self,
               num_heads,
               key_dim,
               dropout=0.0,
               **kwargs):
        super(MyMultiHeadAttention, self).__init__(**kwargs)
        
        self._key_dim = key_dim
        self._num_heads = num_heads
        self._dropout = dropout
        
    def build(self, input_shape):

        self._query_dense = EinsumDense(
            'abcd,def->abcef',
            output_shape=[None,None,self._num_heads,self._key_dim],
            bias_axes='ef',
            name="query",
        )
        
        self._key_dense = EinsumDense(
            'abcd,def->abcef',
            output_shape=[None,None,self._num_heads,self._key_dim],
            bias_axes='ef',
            name="key",
        )
        
        self._value_dense = EinsumDense(
            'abcd,def->abcef',
            output_shape=[None,None,self._num_heads,self._key_dim],
            bias_axes='ef',
            name="value",
        )
        
        self._output_dense = EinsumDense(
            'abcde,def->abcf',
            output_shape=[None,None,self._key_dim],
            bias_axes='f',
            name="attention_output",
        )
        
        self._dropout_layer = layers.Dropout(rate=self._dropout)
    
    def get_config(self):
        config = {
            '_key_dim': self._key_dim,
            '_num_heads': self._num_heads,
            '_dropout': self._dropout,
        }
        base_config = super(MyMultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def _compute_attention(self,
                         query,
                         key,
                         value,
                         training,):
        
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        query = tf.keras.layers.Permute([3,1,2,4])(query)
        key = tf.keras.layers.Permute([3,1,4,2])(key)
        value = tf.keras.layers.Permute([3,1,2,4])(value)
        
        attention_scores = tf.matmul(query, key)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        # print(attention_scores.shape)
        attention_scores_dropout = self._dropout_layer(
            attention_scores,training=training)
        
        
        attention_output = tf.matmul(attention_scores_dropout, value)
        
        attention_output = tf.keras.layers.Permute([2,3,1,4])(attention_output)
        
        return attention_output, attention_scores
    
    def call(self, inputs, training=None):
        query = self._query_dense(inputs)
        key = self._key_dense(inputs)
        value = self._value_dense(inputs)
        
        attention_output, attention_scores = self._compute_attention(
            query, key, value, training
        )
        
        
        
        attention_output = self._output_dense(attention_output)

        
        return attention_output



class SqueezeExcitation(tf.keras.layers.Layer):
  """Creates a squeeze and excitation layer."""

  def __init__(self,
               in_filters,
               out_filters,
               se_ratio,
               **kwargs):

    super(SqueezeExcitation, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio

  def build(self, input_shape):
    num_reduced_filters = max(1, int(self._in_filters * self._se_ratio))
    self._se_reduce = tf.keras.layers.Conv2D(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        )

    self._se_expand = tf.keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        )

    super(SqueezeExcitation, self).build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'se_ratio': self._se_ratio,
    }
    base_config = super(SqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    x = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    x = tf.nn.swish(self._se_reduce(x))
    x = tf.nn.sigmoid(self._se_expand(x))
    return x * inputs



def conv_block(x, filters=16, kernel_size=3, strides=2):
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same"
    )(x)
    x = layers.BatchNormalization(momentum=0.1)(x)
    x = tf.nn.swish(x)
    return x


# Reference: https://git.io/JKgtC


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization(momentum=0.1)(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D(padding=correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization(momentum=0.1)(m)
    m = tf.nn.swish(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization(momentum=0.1)(m)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = MyMultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.1,)
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features



def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """
    img_dim = 2 if tf.keras.backend.image_data_format() == "channels_first" else 1
    input_size = tf.keras.backend.int_shape(inputs)[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )
    

def create_mobilevit(num_classes=1000):
    inputs = keras.Input((image_size, image_size, 3))
    # x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Initial conv-stem -> MV2 block.
    x = conv_block(inputs, filters=16)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=32
    )

    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=32 * expansion_factor, output_channels=48, strides=2
    )
    x = inverted_residual_block(
        x, expanded_channels=48 * expansion_factor, output_channels=48
    )
    x = inverted_residual_block(
        x, expanded_channels=48 * expansion_factor, output_channels=48
    )

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=48 * expansion_factor, output_channels=64, strides=2
    )
    # x = SqueezeExcitation(48, 48, 0.25)(x)
    x = mobilevit_block(x, num_blocks=2, projection_dim=96)
    

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2
    )
    # x = SqueezeExcitation(64, 64, 0.25)(x)
    x = mobilevit_block(x, num_blocks=4, projection_dim=120)
    

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=96 * expansion_factor, output_channels=96, strides=2
    )
    # x = SqueezeExcitation(80, 80, 0.25)(x)
    x = mobilevit_block(x, num_blocks=3, projection_dim=144)
    x = conv_block(x, filters=384, kernel_size=1, strides=1)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    print("MODEL")
    # main()
    model = create_mobilevit()
    model.summary()
    
    model(tf.ones((2,256,256,3), dtype=tf.bfloat16))
    
