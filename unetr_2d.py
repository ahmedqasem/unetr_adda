import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model


def mlp(x, cf):
    x = L.Dense(cf['mlp_dim'], activation='gelu')(x)
    x = L.Dense(cf["mlp_dim"], activation="gelu")(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    x = L.Dense(cf["hidden_dim"])(x)
    x = L.Dropout(cf["dropout_rate"])(x)

    return x


def transformer_encoder(x, cf):
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(
        num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]
    )(x, x)
    x = L.Add()([x, skip_1])

    skip_2 = x
    x = L.LayerNormalization()(x)
    x = mlp(x, cf)
    x = L.Add()([x, skip_2])
    return x


def conv_block(x, num_filters, kernel_size=3):
    x = L.Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


def deconv_block(x, num_filters):
    x = L.Conv2DTranspose(num_filters, kernel_size=2, padding='same', strides=2)(x)
    return x


def build_unetr_2d(cf):
    # inputs
    input_shape = (cf['num_patches'], cf['patch_size'] * cf['patch_size'] * cf['num_channels'])
    inputs = L.Input(input_shape)  # (None, 256, 768)
    # print('input ', inputs.shape)
    # convert into patch embeddings

    # patch + position embedding
    patch_embed = L.Dense(cf['hidden_dim'])(inputs)  # (None, 256, 768)
    # print('patch_embed ', patch_embed.shape)

    positions = tf.range(start=0, limit=cf['num_patches'], delta=1)  # (256, )
    pos_embed = L.Embedding(input_dim=cf['num_patches'], output_dim=cf['hidden_dim'])(positions)
    # print('pos_embed ', pos_embed.shape)

    x = patch_embed + pos_embed

    ''' Transformer Encoder '''
    skip_connection_index = [3, 6, 9, 12]
    skip_connections = []

    for i in range(1, cf['num_layers'] + 1, 1):
        x = transformer_encoder(x, cf)
        # add the skip connection
        if i in skip_connection_index:
            skip_connections.append(x)

    ''' CNN Decoder '''
    z3, z6, z9, z12 = skip_connections

    size = cf['image_size'] // cf['patch_size']
    z0 = L.Reshape((cf['image_size'], cf['image_size'], cf['num_channels']))(inputs)
    z3 = L.Reshape((size, size, cf['hidden_dim']))(z3)
    z6 = L.Reshape((size, size, cf['hidden_dim']))(z6)
    z9 = L.Reshape((size, size, cf['hidden_dim']))(z9)
    z12 = L.Reshape((size, size, cf['hidden_dim']))(z12)
    # print(z0.shape, z3.shape)

    # Decoder 1
    x = deconv_block(x=z12, num_filters=512)

    s = deconv_block(x=z9, num_filters=512)
    s = conv_block(x=s, num_filters=512)

    x = L.Concatenate()([x, s])

    x = conv_block(x, num_filters=512)
    x = conv_block(x, num_filters=512)

    # Decoder 2
    x = deconv_block(x=x, num_filters=256)  # green

    s = deconv_block(x=z6, num_filters=256)  # blue 1
    s = conv_block(x=s, num_filters=256)  # blue 1
    s = deconv_block(x=s, num_filters=256)  # blue 2
    s = conv_block(x=s, num_filters=256)  # blue 2

    x = L.Concatenate()([x, s])

    x = conv_block(x=x, num_filters=256)  # yellow
    x = conv_block(x=x, num_filters=256)  # yellow

    # Decoder 3
    x = deconv_block(x=x, num_filters=128)  # green

    s = deconv_block(x=z3, num_filters=128)  # blue 1
    s = conv_block(x=s, num_filters=128)  # blue 1
    s = deconv_block(x=s, num_filters=128)  # blue 2
    s = conv_block(x=s, num_filters=128)  # blue 2
    s = deconv_block(x=s, num_filters=128)  # blue 3
    s = conv_block(x=s, num_filters=128)  # blue 3

    x = L.Concatenate()([x, s])

    x = conv_block(x=x, num_filters=128)  # yellow
    x = conv_block(x=x, num_filters=128)  # yellow

    # Decoder 4
    x = deconv_block(x=x, num_filters=64)  # green

    s = conv_block(x=z0, num_filters=64)  # yellow
    s = conv_block(x=s, num_filters=64)   # yellow

    x = L.Concatenate()([x, s])

    x = conv_block(x=x, num_filters=64)  # yellow
    x = conv_block(x=x, num_filters=64)  # yellow

    # Output layer
    outputs = L.Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(x)

    return Model(inputs, outputs, name='UNETR_2D')






if __name__ == '__main__':
    config = dict()
    config["image_size"] = 256
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    config["num_patches"] = 256
    config["patch_size"] = 16
    config["num_channels"] = 3

    model = build_unetr_2d(config)
    model.summary()
