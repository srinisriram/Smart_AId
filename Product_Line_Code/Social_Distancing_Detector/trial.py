import math

def finalDist(firstSide, angle, secondSide):
    thirdSide = math.sqrt((secondSide ** 2) + (firstSide ** 2) - 2 * secondSide * firstSide * math.cos((angle*(math.pi)/180)))
    return round(thirdSide)

angle = 60

rad = angle

print(rad)

print(math.cos(rad))

print(finalDist(80, 60, 80))
