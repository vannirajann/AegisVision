from fence import RestrictedZone

def test_point_in_polygon():
    # A square zone from (150,300) to (450,550)
    zone = RestrictedZone([(150, 300), (450, 300), (450, 550), (150, 550)])

    point_inside = (300, 400)   # clearly in the middle of the square
    point_outside = (50, 50)    # clearly outside, top-left corner area

    result_inside = zone.is_inside(point_inside)
    result_outside = zone.is_inside(point_outside)

    print("Point inside zone?", result_inside)
    print("Point outside zone?", result_outside)

    assert result_inside == True
    assert result_outside == False
    print("✅ Test passed: correctly detected inside vs outside the zone")

if __name__ == "__main__":
    test_point_in_polygon()