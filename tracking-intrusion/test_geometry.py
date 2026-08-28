from geometry import get_center_point

def test_basic_bbox():
    bbox = {"x1": 100, "y1": 120, "x2": 300, "y2": 500}
    point = get_center_point(bbox)
    print("Center point:", point)

    # x1=100, x2=300 -> center_x should be 200
    # y2=500 -> bottom_y should be 500
    assert point == (200.0, 500), f"Unexpected result: {point}"
    print("✅ Test passed!")

if __name__ == "__main__":
    test_basic_bbox()