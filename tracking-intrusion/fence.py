import time


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
        Returns "left", "right", or "on_line" depending on which side
        of the line 'point' falls on, using the cross product.
        """
        ax, ay = self.point_a
        bx, by = self.point_b
        px, py = point

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
        self.last_side = {}  # track_id -> "left" or "right"

    def check_crossing(self, track_id, point):
        current_side = self.line.side_of_line(point)
        crossed = False

        if track_id in self.last_side:
            previous_side = self.last_side[track_id]
            if previous_side != current_side and previous_side != "on_line" and current_side != "on_line":
                crossed = True

        self.last_side[track_id] = current_side
        return crossed


class RestrictedZone:
    """
    Represents a restricted area as a polygon (list of points forming a closed shape).
    Used to detect whether a tracked object has entered the zone.
    """

    def __init__(self, polygon_points):
        self.polygon_points = polygon_points

    def is_inside(self, point):
        """
        Ray casting algorithm: checks if 'point' is inside the polygon.
        """
        x, y = point
        n = len(self.polygon_points)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.polygon_points[i]
            xj, yj = self.polygon_points[j]

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
    Also applies a cooldown so the same object can't re-trigger an
    alert too rapidly (e.g. from tracking flicker).
    """

    def __init__(self, restricted_zone, cooldown_seconds=5):
        self.zone = restricted_zone
        self.last_state = {}
        self.last_alert_time = {}
        self.cooldown_seconds = cooldown_seconds

    def check_entry(self, track_id, point):
        currently_inside = self.zone.is_inside(point)
        current_state = "inside" if currently_inside else "outside"

        entered = False

        if track_id in self.last_state:
            previous_state = self.last_state[track_id]
            if previous_state == "outside" and current_state == "inside":
                now = time.time()
                last_time = self.last_alert_time.get(track_id, 0)
                if now - last_time >= self.cooldown_seconds:
                    entered = True
                    self.last_alert_time[track_id] = now

        self.last_state[track_id] = current_state
        return entered