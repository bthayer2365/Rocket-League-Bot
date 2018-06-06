

screen_width = 640
screen_height = 480


def get_dims(style=0):
    # height, width

    if style == 1:
        ball_dims = (160, 120)
        ball_start = 170
        car_dims = (160, 192)
        car_start = 294
    elif style == 2:
        ball_dims = (60, 400)
        ball_start = screen_height - ball_dims[0]  # bottom of screen
        car_dims = (200, 256)
        car_start = 220
    else:
        ball_dims = (80, 96)
        ball_start = 170
        car_dims = (90, 120)
        car_start = 325

    return ball_dims, ball_start, car_dims, car_start


def get_screen_bbox():
    return 640, 300, 640+640, 300+480


def get_bbox(height, width, y_start, x_start=None):
    # Assumes bbox is centered across x
    y_end = y_start + height

    if x_start is None:
        x_start = screen_width // 2 - width // 2
    x_end = x_start + width

    return x_start, y_start, x_end, y_end


def get_ball_bbox(style=0):
    ball_dims, ball_start, car_dims, car_start = get_dims(style)
    return get_bbox(*ball_dims, ball_start)


def get_car_bbox(style=0):
    ball_dims, ball_start, car_dims, car_start = get_dims(style)
    return get_bbox(*car_dims, car_start)


def get_bboxes(style=0):
    ball_dims, ball_start, car_dims, car_start = get_dims(style)
    return get_bbox(*ball_dims, ball_start), get_bbox(*car_dims, car_start)
