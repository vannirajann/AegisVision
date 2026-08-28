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
class RestrictedZone:
    """
    Represents a restricted area as a polygon (list of points forming a closed shape).
    Used to detect whether a tracked object has entered the zone.
    """

    def __init__(self, polygon_points):
        self.polygon_points = polygon_points  # e.g. [(150,300),(450,300),(450,550),(150,550)]

    def is_inside(self, point):
        """
        Ray casting algorithm: checks if 'point' is inside the polygon.
        Returns True if inside, False if outside.
        """
        x, y = point
        n = len(self.polygon_points)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.polygon_points[i]
            xj, yj = self.polygon_points[j]

            # Check if the horizontal ray from 'point' crosses this polygon edge
            intersects = ((yi > y) != (yj > y)) and \
                         (x < (xj - xi) * (y - yi) / (yj - yi) + xi)

            if intersects:
                inside = not inside

            j = i

        return inside
class ZoneEntryDetector:
    """
    Tracks whether each object was inside or outside the restricted zone
    last frame, so we can detect the exact moment it ENTERS.
    """

    def __init__(self, restricted_zone):
        self.zone = restricted_zone
        self.last_state = {}  # track_id -> "inside" or "outside"

    def check_entry(self, track_id, point):
        """
        Call this once per object, per frame, with its current point.
        Returns True only on the frame the object goes from outside -> inside.
        """
        currently_inside = self.zone.is_inside(point)
        current_state = "inside" if currently_inside else "outside"

        entered = False

        if track_id in self.last_state:
            previous_state = self.last_state[track_id]
            if previous_state == "outside" and current_state == "inside":
                entered = True

        self.last_state[track_id] = current_state

        return entered