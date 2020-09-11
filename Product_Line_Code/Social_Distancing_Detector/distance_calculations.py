import math
from constants import cam_focal_point

def calcDistance(x):
    y = cam_focal_point / x * 12
    # y = 275 / x * 12
    y = round(y)
    return y


def get_angle(a, b, c):
    angle1 = (abs(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])))
    angle2 = (abs(math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))))
    return round(angle1, 0), round(angle2, 0)


def finalDist(firstSide, angle, secondSide):
    thirdSide = math.sqrt((secondSide ** 2) + (firstSide ** 2) - 2 * secondSide * firstSide * math.cos((angle * (math.pi) / 180 )))
    return round(thirdSide)
