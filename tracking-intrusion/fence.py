class VirtualLine:
    """
    Represents a virtual fence: a straight line between two points.
    Used to detect which side of the line an object is on,
    and later, whether it has crossed from one side to the other.
    """

    def __init__(self, point_a, point_b):
        self.point_a = point_a  # e.g. (100, 400)
        self.point_b = point_b  # e.g. (500, 400)

    def side_of_line(self, point):
        """
        Returns a positive number if 'point' is on one side of the line,
        a negative number if it's on the other side,
        and 0 if it's exactly ON the line.

        This uses the "cross product" trick: it compares the direction
        from A->B against the direction from A->point. The sign tells
        us which side point falls on.
        """
        ax, ay = self.point_a
        bx, by = self.point_b
        px, py = point

        # Cross product of vector AB and vector AP
        cross_product = (bx - ax) * (py - ay) - (by - ay) * (px - ax)

        if cross_product > 0:
            return "left"
        elif cross_product < 0:
            return "right"
        else:
            return "on_line"