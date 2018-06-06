from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, Activation, Dense, MaxPool2D, Flatten, concatenate
from keras.optimizers import SGD


def get_c3d3_relu():
    model_car_input = Input(shape=(88, 120, 3))
    model_car = Dropout(0.8)(model_car_input)

    # Car model has 8408 parameters
    model_car = Conv2D(8, [5, 5], strides=[1, 1])(model_car)
    model_car = Activation('relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 44 x 60
    model_car = Dropout(0.5)(model_car)

    model_car = Conv2D(8, [5, 5], strides=[1, 1])(model_car)
    model_car = Activation('relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 22 x 30
    model_car = Dropout(0.5)(model_car)

    model_car = Conv2D(8, [3, 3], strides=[1, 1])(model_car)
    model_car = Activation('relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 11 x 15
    model_car = Dropout(0.5)(model_car)

    model_car = Flatten()(model_car)

    model_ball_input = Input(shape=(160, 128, 3))
    model_ball = Dropout(0.8)(model_ball_input)

    model_ball = Conv2D(8, [5, 5], strides=[1, 1])(model_ball)
    model_ball = Activation('relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 80 x 64
    model_ball = Dropout(0.5)(model_ball)

    model_ball = Conv2D(8, [5, 5], strides=[1, 1])(model_ball)
    model_ball = Activation('relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 40 x 32
    model_ball = Dropout(0.5)(model_ball)

    model_ball = Conv2D(8, [3, 3], strides=[1, 1])(model_ball)
    model_ball = Activation('relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 20 x 16
    model_ball = Dropout(0.5)(model_ball)

    model_ball = Flatten()(model_ball)

    model_combined = concatenate([model_car, model_ball])

    model_combined = Dense(750)(model_combined)
    model_combined = Activation('relu')(model_combined)
    model_combined = Dropout(0.5)(model_combined)

    model_combined = Dense(100)(model_combined)
    model_combined = Activation('relu')(model_combined)
    model_combined = Dropout(0.5)(model_combined)

    model_combined = Dense(3)(model_combined)
    model_combined = Activation('softmax')(model_combined)

    model = Model(inputs=[model_car_input, model_ball_input], outputs=model_combined)
    sgd = SGD(0.001)
    model.compile(sgd, 'categorical_crossentropy')

    return model


def get_c3d2_sigmoid():
    model_car_input = Input(shape=(90, 120, 3))
    model_car = Dropout(0.8)(model_car_input)

    # Car model has 8408 parameters
    model_car = Conv2D(8, [5, 5], strides=[1, 1])(model_car)
    model_car = Activation('relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 45 x 60
    model_car = Dropout(0.5)(model_car)

    model_car = Conv2D(8, [5, 5], strides=[1, 1])(model_car)
    model_car = Activation('relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 22 x 30
    model_car = Dropout(0.5)(model_car)

    model_car = Conv2D(8, [3, 3], strides=[1, 1])(model_car)
    model_car = Activation('relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 11 x 15
    model_car = Dropout(0.5)(model_car)

    model_car = Flatten()(model_car)

    model_ball_input = Input(shape=(160, 125, 3))
    model_ball = Dropout(0.8)(model_ball_input)

    model_ball = Conv2D(8, [5, 5], strides=[1, 1])(model_ball)
    model_ball = Activation('relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 80 x 62
    model_ball = Dropout(0.5)(model_ball)

    model_ball = Conv2D(8, [5, 5], strides=[1, 1])(model_ball)
    model_ball = Activation('relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 40 x 31
    model_ball = Dropout(0.5)(model_ball)

    model_ball = Conv2D(8, [3, 3], strides=[1, 1])(model_ball)
    model_ball = Activation('relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 20 x 15
    model_ball = Dropout(0.5)(model_ball)

    model_ball = Flatten()(model_ball)

    model_combined = concatenate([model_car, model_ball])

    model_combined = Dense(30)(model_combined)
    model_combined = Activation('sigmoid')(model_combined)
    model_combined = Dropout(0.5)(model_combined)

    model_combined = Dense(3)(model_combined)
    model_combined = Activation('softmax')(model_combined)

    model = Model(inputs=[model_car_input, model_ball_input], outputs=model_combined)
    sgd = SGD(0.001)
    model.compile(sgd, 'categorical_crossentropy')

    return model


def get_c1d2():
    model_car_input = Input(shape=(90, 120, 3))
    model_car = Dropout(0.8)(model_car_input)

    # Car model has 8408 parameters
    model_car = Conv2D(8, [5, 5], strides=[1, 1])(model_car)
    model_car = Activation('relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 45 x 60
    model_car = Dropout(0.5)(model_car)

    model_car = Flatten()(model_car)

    model_ball_input = Input(shape=(160, 125, 3))
    model_ball = Dropout(0.8)(model_ball_input)

    model_ball = Conv2D(8, [5, 5], strides=[1, 1])(model_ball)
    model_ball = Activation('relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 80 x 62
    model_ball = Dropout(0.5)(model_ball)

    model_ball = Flatten()(model_ball)

    model_combined = concatenate([model_car, model_ball])

    model_combined = Dense(30)(model_combined)
    model_combined = Activation('sigmoid')(model_combined)
    model_combined = Dropout(0.5)(model_combined)

    model_combined = Dense(3)(model_combined)
    model_combined = Activation('softmax')(model_combined)

    model = Model(inputs=[model_car_input, model_ball_input], outputs=model_combined)
    sgd = SGD(0.001)
    model.compile(sgd, 'categorical_crossentropy')

    return model


def get_d3():
    model_car_input = Input(shape=(90, 120, 3))
    model_car = Dropout(0.8)(model_car_input)

    model_car = Flatten()(model_car)

    model_ball_input = Input(shape=(160, 125, 3))
    model_ball = Dropout(0.8)(model_ball_input)

    model_ball = Flatten()(model_ball)

    model_combined = concatenate([model_car, model_ball])

    model_combined = Dense(750)(model_combined)
    model_combined = Activation('relu')(model_combined)
    model_combined = Dropout(0.5)(model_combined)

    model_combined = Dense(100)(model_combined)
    model_combined = Activation('relu')(model_combined)
    model_combined = Dropout(0.5)(model_combined)

    model_combined = Dense(3)(model_combined)
    model_combined = Activation('softmax')(model_combined)

    model = Model(inputs=[model_car_input, model_ball_input], outputs=model_combined)
    sgd = SGD(0.001)
    model.compile(sgd, 'categorical_crossentropy')

    return model


def get_gan():

    # car input
    # ball input
    # random input

    # output

    # Blur the data so it's much more of an average
    # After training, we should get situations that are much more
    # Image to vec
    pass


get_model = get_c3d2_sigmoid
