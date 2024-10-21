import math
import tensorflow as tf
from tensorflow import keras

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

class AttentionBlock(keras.layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = keras.layers.GroupNormalization(groups=groups)
        self.query = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = keras.layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("blc, bLc->blL", q, k) * scale
        attn_score = tf.nn.softmax(attn_score, -1)

        proj = tf.einsum("blL,bLc->blc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

class PositionalEncoding1D(keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        super(PositionalEncoding1D, self).__init__()
        
        self.channels = int(2 * tf.math.ceil(channels / 2))
        self.inv_freq = 1 / tf.math.pow(10000, tf.range(0, self.channels, 2, dtype=dtype) / self.channels)
        self.cached_penc = None

    @tf.function
    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
        emb = tf.reshape(emb, (*emb.shape[:-2], -1))
        return emb

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(inputs.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, org_channels = inputs.shape

        dtype = self.inv_freq.dtype
        pos_x = tf.range(x, dtype=dtype)
        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = self.get_emb(sin_inp_x)

        cached_penc = tf.repeat(
            emb[None, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        return cached_penc

def ResidualBlock_Down(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[-1]

        if input_width == width:
            residual = x
        else:
            residual = keras.layers.Conv1D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = keras.layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, :]

        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x = keras.layers.Add()([x, temb])
        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = keras.layers.Add()([x, residual])
        return x

    return apply

def ResidualBlock_Up(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t, c = inputs
        input_width = x.shape[-1]

        if input_width == width:
            residual = x
        else:
            residual = keras.layers.Conv1D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = keras.layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, :]
            
        cemb = activation_fn(c)
        cemb = keras.layers.Dense(width, kernel_initializer=kernel_init(1.0))(cemb)[:, None, :]

        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x = keras.layers.Add()([cemb * x, temb])
        x = keras.layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = keras.layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = keras.layers.Conv1D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = keras.layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply
    
def TransformerEncoder(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        pemb = keras.layers.Reshape((-1, 100))(inputs)
        pemb = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(pemb)

        pemb = AttentionBlock(units, groups=8)(pemb)
        pemb = keras.layers.GlobalAveragePooling1D()(pemb)
        pemb = keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(pemb)
        return pemb
    
    return apply

def build_model(
    sequence_length,
    widths,
    has_attention,
    capacity_matrix_size=(100,100,1),
    first_conv_channels=8,
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.gelu,
):
    
    image_input = keras.layers.Input(shape=(sequence_length, 1), name="input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    capacity_matrix_input = keras.Input(shape=capacity_matrix_size, name="capacity_matrix_input")
    condition_mask = keras.Input(shape=(first_conv_channels * 4,), name="mask_input")

    x = keras.layers.Conv1D(
        first_conv_channels,
        kernel_size=7,
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    posemb = PositionalEncoding1D(first_conv_channels)(x)
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    cemb = TransformerEncoder(units=first_conv_channels * 4, activation_fn=activation_fn)(capacity_matrix_input)
    
    x = keras.layers.Concatenate()([x, posemb])
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock_Down(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock_Down(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock_Down(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = keras.layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock_Up(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb, condition_mask * cemb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = keras.layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = keras.layers.Conv1D(1, kernel_size=7, padding="same", kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([image_input, time_input, capacity_matrix_input, condition_mask], x, name="unet")