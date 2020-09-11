import RPi.GPIO as gpio
from constants import MOTOR1_FORWARD_GPIO, ON, OFF
import time


class OpenDoor:
    """
    This class will implement the logic to open the door for both mask and attendance tracking.
    """

    def __init__(self):
        self.setup_gpio()

    def setup_gpio(self):
        gpio.setmode(gpio.BCM)
        gpio.setwarnings(False)
        gpio.setup(MOTOR1_FORWARD_GPIO, gpio.OUT)
        gpio.output(MOTOR1_FORWARD_GPIO, OFF)

    def start_motor(self):
        """
        This method will use the Raspberry Pi GPIO PINS to start the motor to open the door.
        """
        print("[INFO]: Opening Door...")
        gpio.output(MOTOR1_FORWARD_GPIO, ON)

    def stop_motor(self):
        print("[INFO]: Stopping Motor...")
        gpio.output(MOTOR1_FORWARD_GPIO, OFF)

    def thread_for_opening_door(self):
        self.start_motor()
        time.sleep(3)
        self.stop_motor()
        time.sleep(2)

