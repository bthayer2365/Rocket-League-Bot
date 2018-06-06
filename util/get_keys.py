# Citation: Box Of Hats (https://github.com/Box-Of-Hats ), Sentdex (https://github.com/Sentdex)

import win32api as wapi

left_mouse = 1
right_mouse = 2
space = 0x20
left_shift = 0xA0

v_keys = [left_mouse, right_mouse, space, left_shift]
k_keys = "WASD0123456789"

key_list = [ord(c) for c in list(k_keys)] + v_keys


def key_check():
    keys = []
    for key in key_list:
        if wapi.GetAsyncKeyState(key):
            keys.append(chr(key))
    return keys
