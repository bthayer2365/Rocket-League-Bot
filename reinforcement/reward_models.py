from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPool2D, Flatten, Dense, concatenate

from keras.optimizers import SGD


def create_reward_func(model, ball_dims, car_dims):
    def reward_func(obs):
        ball, car = obs
        ball = ball.reshape((1, ) + ball_dims)
        car = car.reshape((1, ) + car_dims)
        return 0 if model.predict([ball, car]) < 0.5 else 1
    return reward_func


def get_model_A(ball_dims, car_dims):
    ball_input = Input(ball_dims)
    car_input = Input(car_dims)
    
    ball = Conv2D(8, [5, 5])(ball_input)
    ball = Activation('relu')(ball)
    ball = MaxPool2D()(ball)
    
    ball = Conv2D(16, [3, 3])(ball)
    ball = Activation('relu')(ball)
    ball = MaxPool2D()(ball)
    
    car = Conv2D(8, [5, 5])(car_input)
    car = Activation('relu')(car)
    car = MaxPool2D()(car)
    
    car = Conv2D(16, [3, 3])(car)
    car = Activation('relu')(car)
    car = MaxPool2D()(car)

    ball = Flatten()(ball)
    car = Flatten()(car)

    combined = concatenate([ball, car])

    combined = Dense(25)(combined)
    combined = Activation('relu')(combined)

    combined = Dense(3)(combined)
    combined = Activation('relu')(combined)

    out = Dense(1)(combined)
    out = Activation('sigmoid')(out)

    model = Model(inputs=[ball_input, car_input], outputs=[out])
    optimizer = SGD(0.0001)
    model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return model


def get_model_B(ball_dims, car_dims):
    ball_input = Input(ball_dims)
    car_input = Input(car_dims)

    ball = Conv2D(8, [5, 5])(ball_input)
    ball = Activation('relu')(ball)
    ball = MaxPool2D()(ball)

    ball = Conv2D(16, [3, 3])(ball)
    ball = Activation('relu')(ball)

    ball = Conv2D(8, [3, 3])(ball)
    ball = Activation('relu')(ball)
    ball = MaxPool2D()(ball)

    car = Conv2D(8, [5, 5])(car_input)
    car = Activation('relu')(car)
    car = MaxPool2D()(car)

    car = Conv2D(16, [3, 3])(car)
    car = Activation('relu')(car)

    car = Conv2D(8, [3, 3])(car)
    car = Activation('relu')(car)
    car = MaxPool2D()(car)

    ball = Flatten()(ball)
    car = Flatten()(car)

    combined = concatenate([ball, car])

    combined = Dense(25)(combined)
    combined = Activation('relu')(combined)

    combined = Dense(3)(combined)
    combined = Activation('relu')(combined)

    out = Dense(1)(combined)
    out = Activation('sigmoid')(out)

    model = Model(inputs=[ball_input, car_input], outputs=[out])
    optimizer = SGD(0.0001)
    model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return model


def get_model_C(ball_dims, car_dims):
    ball_input = Input(ball_dims)
    car_input = Input(car_dims)

    ball = Conv2D(8, [5, 5], strides=[2, 2])(ball_input)
    ball = Activation('relu')(ball)

    ball = Conv2D(16, [3, 3])(ball)
    ball = Activation('relu')(ball)
    ball = MaxPool2D()(ball)

    car = Conv2D(8, [5, 5], strides=[2, 2])(car_input)
    car = Activation('relu')(car)

    car = Conv2D(16, [3, 3])(car)
    car = Activation('relu')(car)
    car = MaxPool2D()(car)

    ball = Flatten()(ball)
    car = Flatten()(car)

    combined = concatenate([ball, car])

    combined = Dense(25)(combined)
    combined = Activation('relu')(combined)

    combined = Dense(3)(combined)
    combined = Activation('relu')(combined)

    out = Dense(1)(combined)
    out = Activation('sigmoid')(out)

    model = Model(inputs=[ball_input, car_input], outputs=[out])
    optimizer = SGD(0.0001)
    model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return model


def get_model_D(ball_dims, car_dims):
    ball_input = Input(ball_dims)
    car_input = Input(car_dims)

    ball = Conv2D(8, [5, 5], strides=[2, 2])(ball_input)
    ball = Activation('relu')(ball)

    ball = Conv2D(16, [3, 3], strides=[2, 2])(ball)
    ball = Activation('relu')(ball)

    car = Conv2D(8, [5, 5], strides=[2, 2])(car_input)
    car = Activation('relu')(car)

    car = Conv2D(16, [3, 3], strides=[2, 2])(car)
    car = Activation('relu')(car)

    ball = Flatten()(ball)
    car = Flatten()(car)

    combined = concatenate([ball, car])

    combined = Dense(25)(combined)
    combined = Activation('relu')(combined)

    combined = Dense(3)(combined)
    combined = Activation('relu')(combined)

    out = Dense(1)(combined)
    out = Activation('sigmoid')(out)

    model = Model(inputs=[ball_input, car_input], outputs=[out])
    optimizer = SGD(0.0001)
    model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return model


def get_model_E(ball_dims, car_dims):
    ball_input = Input(ball_dims)
    car_input = Input(car_dims)

    ball = Conv2D(8, [5, 5], strides=[2, 2])(ball_input)
    ball = Activation('relu')(ball)

    ball = Conv2D(16, [3, 3], strides=[2, 2])(ball)
    ball = Activation('relu')(ball)

    car = Conv2D(8, [5, 5], strides=[2, 2])(car_input)
    car = Activation('relu')(car)

    car = Conv2D(16, [3, 3], strides=[2, 2])(car)
    car = Activation('relu')(car)

    ball = Flatten()(ball)
    car = Flatten()(car)

    combined = concatenate([ball, car])

    combined = Dense(25)(combined)
    combined = Activation('sigmoid')(combined)

    out = Dense(1)(combined)
    out = Activation('sigmoid')(out)

    model = Model(inputs=[ball_input, car_input], outputs=[out])
    optimizer = SGD(0.0001)
    model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return model


def get_model_F(ball_dims, car_dims):
    ball_input = Input(ball_dims)
    car_input = Input(car_dims)

    ball = Conv2D(8, [5, 5], strides=[2, 2])(ball_input)
    ball = Activation('relu')(ball)

    ball = Conv2D(16, [3, 3], strides=[2, 2])(ball)
    ball = Activation('relu')(ball)

    car = Conv2D(8, [5, 5], strides=[2, 2])(car_input)
    car = Activation('relu')(car)

    car = Conv2D(16, [3, 3], strides=[2, 2])(car)
    car = Activation('relu')(car)

    ball = Flatten()(ball)
    car = Flatten()(car)

    combined = concatenate([ball, car])

    combined = Dense(25)(combined)
    combined = Activation('relu')(combined)

    out = Dense(1)(combined)
    out = Activation('sigmoid')(out)

    model = Model(inputs=[ball_input, car_input], outputs=[out])
    optimizer = SGD(0.0001)
    model.compile(optimizer, 'binary_crossentropy', metrics=['accuracy'])

    return model
