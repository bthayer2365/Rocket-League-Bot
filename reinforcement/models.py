from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, concatenate
from keras.optimizers import RMSprop, SGD


def get_model_A(ball_dims, car_dims):
    # Ball
    model_ball_input = Input(shape=ball_dims)  # 80 x 96
    model_ball = Conv2D(8, [3, 3], activation='relu')(model_ball_input)
    model_ball = MaxPool2D()(model_ball)  # 40 x 48
    model_ball = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_ball)  # 20 x 24
    model_ball = Conv2D(8, [3, 3], strides=[1, 1], activation='relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 10 x 14

    # Car
    model_car_input = Input(shape=car_dims)
    model_car = Conv2D(8, [3, 3], activation='relu')(model_car_input)  # 90 x 120
    model_car = MaxPool2D()(model_car)  # 44 x 60
    model_car = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_car)  # 22 x 30
    model_car = Conv2D(8, [3, 3], strides=[1, 1], activation='relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 11 x 15

    model_ball = Flatten()(model_ball)  # 140
    model_car = Flatten()(model_car)  # 165
    model_combined = concatenate([model_car, model_ball])  # 305
    model_combined = Dense(25, activation='relu')(model_combined)
    action_values = Dense(3)(model_combined)

    model = Model(inputs=[model_ball_input, model_car_input], outputs=action_values)
    optimizer = RMSprop(0.001)
    model.compile(optimizer, 'mean_squared_error')

    return model


def get_model_B(ball_dims, car_dims):
    # Ball
    model_ball_input = Input(shape=ball_dims)  # 80 x 96
    model_ball = Conv2D(8, [3, 3], activation='relu')(model_ball_input)
    model_ball = MaxPool2D()(model_ball)  # 40 x 48
    model_ball = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_ball)  # 20 x 24
    model_ball = Conv2D(8, [3, 3], strides=[1, 1], activation='relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 10 x 14

    # Car
    model_car_input = Input(shape=car_dims)
    model_car = Conv2D(8, [3, 3], activation='relu')(model_car_input)  # 90 x 120
    model_car = MaxPool2D()(model_car)  # 44 x 60
    model_car = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_car)  # 22 x 30
    model_car = Conv2D(8, [3, 3], strides=[1, 1], activation='relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 11 x 15

    model_ball = Flatten()(model_ball)  # 140
    model_car = Flatten()(model_car)  # 165
    model_combined = concatenate([model_car, model_ball])  # 305
    model_combined = Dense(25, activation='relu')(model_combined)
    action_values = Dense(3, activation='tanh')(model_combined)

    model = Model(inputs=[model_ball_input, model_car_input], outputs=action_values)
    optimizer = SGD(0.001)
    model.compile(optimizer, 'mean_squared_error')

    return model


def get_model_C(ball_dims, car_dims):
    # Ball
    model_ball_input = Input(shape=ball_dims)  # 80 x 96
    model_ball = Conv2D(8, [3, 3], activation='relu')(model_ball_input)
    model_ball = MaxPool2D()(model_ball)  # 40 x 48
    model_ball = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_ball)  # 20 x 24
    model_ball = Conv2D(8, [3, 3], strides=[1, 1], activation='relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 10 x 14

    # Car
    model_car_input = Input(shape=car_dims)
    model_car = Conv2D(8, [3, 3], activation='relu')(model_car_input)  # 90 x 120
    model_car = MaxPool2D()(model_car)  # 44 x 60
    model_car = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_car)  # 22 x 30
    model_car = Conv2D(8, [3, 3], strides=[1, 1], activation='relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 11 x 15

    model_ball = Flatten()(model_ball)  # 140
    model_car = Flatten()(model_car)  # 165
    model_combined = concatenate([model_car, model_ball])  # 305
    model_combined = Dense(25, activation='relu')(model_combined)
    action_values = Dense(3)(model_combined)

    model = Model(inputs=[model_ball_input, model_car_input], outputs=action_values)
    optimizer = SGD(0.001)
    model.compile(optimizer, 'mean_squared_error')

    return model


def get_model_D(ball_dims, car_dims):
    # Ball
    model_ball_input = Input(shape=ball_dims)  # 80 x 96
    model_ball = Conv2D(8, [5, 5], activation='relu')(model_ball_input)
    model_ball = MaxPool2D()(model_ball)  # 40 x 48
    model_ball = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_ball)  # 20 x 24
    model_ball = Conv2D(8, [3, 3], strides=[1, 1], activation='relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 10 x 14

    # Car
    model_car_input = Input(shape=car_dims)
    model_car = Conv2D(8, [5, 5], activation='relu')(model_car_input)  # 90 x 120
    model_car = MaxPool2D()(model_car)  # 44 x 60
    model_car = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_car)  # 22 x 30
    model_car = Conv2D(8, [3, 3], strides=[1, 1], activation='relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 11 x 15

    model_ball = Flatten()(model_ball)  # 140
    model_car = Flatten()(model_car)  # 165
    model_combined = concatenate([model_car, model_ball])  # 305
    model_combined = Dense(25, activation='relu')(model_combined)
    action_values = Dense(3)(model_combined)

    model = Model(inputs=[model_ball_input, model_car_input], outputs=action_values)
    optimizer = SGD(0.001)
    model.compile(optimizer, 'mean_squared_error')

    return model


def get_model_E(ball_dims, car_dims):
    # Ball
    model_ball_input = Input(shape=ball_dims)  # 80 x 96
    model_ball = Conv2D(8, [5, 5], activation='relu')(model_ball_input)
    model_ball = MaxPool2D()(model_ball)  # 40 x 48
    model_ball = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_ball)  # 20 x 24
    model_ball = Conv2D(32, [3, 3], strides=[1, 1], activation='relu')(model_ball)
    model_ball = MaxPool2D()(model_ball)  # 10 x 14

    # Car
    model_car_input = Input(shape=car_dims)
    model_car = Conv2D(8, [5, 5], activation='relu')(model_car_input)  # 90 x 120
    model_car = MaxPool2D()(model_car)  # 44 x 60
    model_car = Conv2D(16, [5, 5], strides=[2, 2], activation='relu')(model_car)  # 22 x 30
    model_car = Conv2D(32, [3, 3], strides=[1, 1], activation='relu')(model_car)
    model_car = MaxPool2D()(model_car)  # 11 x 15

    model_ball = Flatten()(model_ball)  # 140
    model_car = Flatten()(model_car)  # 165
    model_combined = concatenate([model_car, model_ball])  # 305
    model_combined = Dense(100, activation='relu')(model_combined)
    model_combined = Dense(25, activation='relu')(model_combined)
    action_values = Dense(3)(model_combined)

    model = Model(inputs=[model_ball_input, model_car_input], outputs=action_values)
    optimizer = SGD(0.001)
    model.compile(optimizer, 'mean_squared_error')

    return model
