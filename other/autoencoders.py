
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash, z_mean, z_log_var):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        # TODO This assumes shape[0], shape[1] = rows, cols
        # TODO Probably need this to be MSE since MNIST is just binary
        xent_loss = K.shape(x)[0] * K.shape(x)[1] * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        # KL is "distance from uniform"
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # Everything uses the backend to compute, so it finds derivatives automatically
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)  # When this happens, it knows that there's loss to it
        # We don't use this output.
        return x


def sampling(args):  # Generate a random sample
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]),
                              mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon


def vae(shape):
    # What inputs do we want?
    # Input dimensions
    # Hidden vector units
    # Number of convolutional layers, probably 4 or 5. This uses 4, so there's probably more complexity and we should use more
    # Start with 4 then bump it up to 6, although the whole thing ought to be variable
    # Difference in input sizes: 28x28 vs 125x160 and 90x120
    # Need an additional 2 up/downsampling (minimum)
    # Oringinal Structure: CDCCihiuCCU (capitals are convs)
    # My structure: DCDCDCihiuCUCUCU
    # Down(64x80,45x60) -> Conv -> Down(32x40,22x30) -> Conv -> Down(16x20,11x15) -> Conv
    # Flatten -> intermediate -> hidden -> intermediate(up) -> ConvSize(16x20, 11x15)
    # Conv -> Up(32x40,22x30) -> Conv -> Up(64x80,44x60) -> Conv -> Up(128x160,88x120)
    
    
    latent_dim = 2
    intermediate_dim = 128

    # number of convolutional filters to use
    filters = 64
    # convolution kernel size
    num_conv = 3

    down_shape = (shape[0]//8, shape[1]//8, filters)  # Shape after 3 downsamplings

    x = Input(shape)
    conv_1 = Conv2D(shape[2], kernel_size=num_conv, padding='same', activation='relu', strides=2)(x)
    conv_2 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_1)
    conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=2)(conv_2)
    conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_3)
    conv_5 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=2)(conv_4)
    conv_6 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_5)

    flat = Flatten()(conv_6)  # Flatten
    intermediate = Dense(intermediate_dim, activation='relu')(flat)

    # Latent space
    z_mean = Dense(latent_dim)(intermediate)  # Learning the mean of the data
    z_log_var = Dense(latent_dim)(intermediate)  # Learning the log variance of the data

    z = Lambda(sampling)([z_mean, z_log_var])  # Sample from latent space

    upsample_1 = Dense(intermediate_dim, activation='relu')  # Decoding back up again
    upsample_2 = Dense(filters * down_shape[0] * down_shape[1], activation='relu')  # To size of 1st conv

    reshape = Reshape(down_shape)
    # TODO What is the difference between same and valid in this context
    deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
    deconv_2 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=2, activation='relu')
    deconv_3 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
    deconv_4 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=2, activation='relu')
    deconv_5 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
    deconv_6 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=2, activation='relu')
    mean_squash = Conv2D(shape[2], kernel_size=2, padding='valid', activation='sigmoid')
    
    # This is the third going up, there was only 1 downsampling layer, so we only need this 1

    # This is all defining the way back up, so you take z, which was sampled then turn that into an image
    hid_decoded = upsample_1(z)
    up_decoded = upsample_2(hid_decoded)
    reshape_decoded = reshape(up_decoded)
    deconv_1_decoded = deconv_1(reshape_decoded)
    deconv_2_decoded = deconv_2(deconv_1_decoded)
    x_decoded_relu = deconv_3(deconv_2_decoded)  # This should be the original image
    x_decoded_mean_squash = mean_squash(x_decoded_relu)  # TODO Learning the mean?

    y = CustomVariationalLayer()([x, x_decoded_mean_squash, z_mean, z_log_var])  # Also implements the loss
    vae = Model(x, y)
    vae.compile(optimizer='rmsprop', loss=None)  # Loss is a part of the custom layer
    vae.summary()

    # train the VAE on MNIST digits
    (x_train, _), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

    print('x_train.shape:', x_train.shape)

    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _hid_decoded = decoder_hid(decoder_input)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)
    _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
    _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
    _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
    _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
    generator = Model(decoder_input, _x_decoded_mean_squash)

    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
