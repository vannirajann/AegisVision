from fence import VirtualLine

def test_side_of_line():
    # A horizontal line from (100, 400) to (500, 400)
    line = VirtualLine((100, 400), (500, 400))

    point_above = (300, 200)   # smaller y = above the line
    point_below = (300, 600)   # larger y = below the line

    side1 = line.side_of_line(point_above)
    side2 = line.side_of_line(point_below)

    print("Point above line:", side1)
    print("Point below line:", side2)

    assert side1 != side2, "Points on opposite sides should return different results"
    print("✅ Test passed: correctly distinguished both sides of the line")

if __name__ == "__main__":
    test_side_of_line()