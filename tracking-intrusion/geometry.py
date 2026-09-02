def get_center_point(bbox):
    """
    Takes a bounding box and returns a single point to track.
    We use the BOTTOM-CENTER of the box, not the true center,
    because for a standing/walking person, the bottom-center is
    roughly where their feet touch the ground — which is what
    matters for line-crossing and zone-entry decisions.
    """
    x1 = bbox["x1"]
    y1 = bbox["y1"]
    x2 = bbox["x2"]
    y2 = bbox["y2"]

    center_x = (x1 + x2) / 2
    bottom_y = y2  # y2 is the bottom edge of the box

    return (center_x, bottom_y)