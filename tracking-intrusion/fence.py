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
class LineCrossingDetector:
    """
    Tracks which side of the line each object was on last frame,
    so we can detect the exact moment an object crosses over.
    """

    def __init__(self, virtual_line):
        self.line = virtual_line
        self.last_side = {}  # track_id -> "left" or "right" (last known side)

    def check_crossing(self, track_id, point):
        """
        Call this once per object, per frame, with its current point.
        Returns True if this object just crossed the line THIS frame,
        otherwise False.
        """
        current_side = self.line.side_of_line(point)

        crossed = False

        if track_id in self.last_side:
            previous_side = self.last_side[track_id]
            # Only count it as a crossing if both sides are valid (not "on_line")
            # and they are DIFFERENT from each other.
            if previous_side != current_side and previous_side != "on_line" and current_side != "on_line":
                crossed = True

        # Remember this side for next frame's comparison
        self.last_side[track_id] = current_side

        return crossed