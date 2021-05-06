import screen_brightness_control as sbc

current_brightness = sbc.get_brightness()

def set_brightness(val):
    try:
        sbc.set_brightness(val)
    except sbc.ScreenBrightnessError as error:
        print(error)
