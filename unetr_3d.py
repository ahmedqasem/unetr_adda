import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from configuration import config
from image_ops import load_sample_image, apply_window, explore_3dimage_interact


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


def deconv_block3d(x, num_filters):
    """Projects the reshaped tensors from the embedding space into the input space at each resolution using a 3D deconvolution layer.
    example: input_tensor = tf.random.normal([1, 1, 32, 32, 32])
    (batch_size, channels, depth, rows, columns)
     Args:
       x: A tensor of shape (batch_size, height, width, channels).

     Returns:
       A tensor of shape (batch_size, height, width, channels).
     """
    x = L.Conv3DTranspose(num_filters,
                          kernel_size=(2, 2, 2),
                          padding='same',
                          strides=(2, 2, 2))(x)
    return x


def conv_block3d(x, num_filters, kernel_size=3):
    x = L.Conv3D(num_filters,
                 kernel_size=(kernel_size, kernel_size, kernel_size),
                 padding='same')(x)
    # x = tf.squeeze(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


def build_unetr_3d(cf):
    # inputs
    input_shape = (cf['num_patches'], cf['patch_size'] * cf['patch_size'] * cf['patch_size'] * cf['num_channels'])
    inputs = L.Input(input_shape)
    print('input ', inputs.shape)

    # patch + position embedding
    patch_embed = L.Dense(cf['hidden_dim'])(inputs)
    print('patch_embed ', patch_embed.shape)

    positions = tf.range(start=0, limit=cf['num_patches'], delta=1)  # (256, )
    pos_embed = L.Embedding(input_dim=cf['num_patches'], output_dim=cf['hidden_dim'])(positions)
    # print(positions)
    print('pos_embed ', pos_embed.shape)

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

    h = cf['image_height'] // cf['patch_size']
    w = cf['image_width'] // cf['patch_size']
    d = cf['image_depth'] // cf['patch_size']

    z0 = L.Reshape((cf['image_height'], cf['image_width'], cf['image_depth'], cf['num_channels']))(inputs)
    z3 = L.Reshape((h, w, d, cf['hidden_dim']))(z3)
    z6 = L.Reshape((h, w, d, cf['hidden_dim']))(z6)
    z9 = L.Reshape((h, w, d, cf['hidden_dim']))(z9)
    z12 = L.Reshape((h, w, d, cf['hidden_dim']))(z12)

    print('reshaping: ', z0.shape, z3.shape, z6.shape, z9.shape, z12.shape)

    cf['image_height']
    # Decoder 1
    x = deconv_block3d(x=z12, num_filters=cf['filters'][0])

    s = deconv_block3d(x=z9, num_filters=cf['filters'][0])
    s = conv_block3d(x=s, num_filters=cf['filters'][0])
    # print(x.shape, s.shape)
    x = L.Concatenate()([x, s])
    # print(x.shape)
    x = conv_block3d(x, num_filters=cf['filters'][0])
    x = conv_block3d(x, num_filters=cf['filters'][0])
    # print(x.shape)

    # Decoder 2
    x = deconv_block3d(x=x, num_filters=cf['filters'][1])  # green

    s = deconv_block3d(x=z6, num_filters=cf['filters'][1])  # blue 1
    s = conv_block3d(x=s, num_filters=cf['filters'][1])  # blue 1
    s = deconv_block3d(x=s, num_filters=cf['filters'][1])  # blue 2
    s = conv_block3d(x=s, num_filters=cf['filters'][1])  # blue 2
    # print(x.shape, s.shape)
    x = L.Concatenate()([x, s])

    x = conv_block3d(x=x, num_filters=cf['filters'][1])  # yellow
    x = conv_block3d(x=x, num_filters=cf['filters'][1])  # yellow
    # print(x.shape)

    # Decoder 3
    x = deconv_block3d(x=x, num_filters=cf['filters'][2])  # green

    s = deconv_block3d(x=z3, num_filters=cf['filters'][2])  # blue 1
    s = conv_block3d(x=s, num_filters=cf['filters'][2])  # blue 1
    s = deconv_block3d(x=s, num_filters=cf['filters'][2])  # blue 2
    s = conv_block3d(x=s, num_filters=cf['filters'][2])  # blue 2
    s = deconv_block3d(x=s, num_filters=cf['filters'][2])  # blue 3
    s = conv_block3d(x=s, num_filters=cf['filters'][2])  # blue 3
    # print(x.shape, s.shape)
    x = L.Concatenate()([x, s])

    x = conv_block3d(x=x, num_filters=cf['filters'][2])  # yellow
    x = conv_block3d(x=x, num_filters=cf['filters'][2])  # yellow
    print(x.shape)

    # Decoder 4
    x = deconv_block3d(x=x, num_filters=cf['filters'][3])  # green
    print(x.shape)

    s = conv_block3d(x=z0, num_filters=cf['filters'][3])  # yellow
    s = conv_block3d(x=s, num_filters=cf['filters'][3])  # yellow
    print(x.shape, s.shape)
    x = L.Concatenate()([x, s])

    x = conv_block3d(x=x, num_filters=cf['filters'][3])  # yellow
    x = conv_block3d(x=x, num_filters=cf['filters'][3])  # yellow
    # Output layer
    outputs = L.Conv3D(1, kernel_size=1, padding='same', activation='sigmoid')(x)

    return Model(inputs, outputs, name='UNETR_3D')


if __name__ == '__main__':



    model = build_unetr_3d(config)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    # load one image
    patches, labels = load_sample_image()
    print(patches.shape, labels.shape)

    # Train the model
    model.fit(patches, labels, epochs=2)

    # img_patches = load_sample_image()
    # y, x, z = 3, 4, 2
    # img = img_patches[x][y][z]
    # # img = tf.reshape(img, [1, 500, 500, 415, 1])
    #
    # # Add the extra dimensions to the image tensor
    # img = tf.expand_dims(img, axis=0)
    # img = tf.expand_dims(img, axis=-1)
    # print(img.shape)
    # output_tensor = conv_block3d(img, 64)
    # print(output_tensor.shape)
    #
    # xx = tf.squeeze(img)
    # yy = tf.squeeze(output_tensor)[:, :, :, 1]
    # print(xx.shape, yy.shape)
    #
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(ncols=2)
    # ax[0].imshow(xx[:, :, 10], cmap='gray')
    # ax[1].imshow(yy[:, :, 10], cmap='gray')
