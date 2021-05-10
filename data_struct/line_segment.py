import math
import matplotlib.pyplot as plt
import numpy as np
from decimal import *

quantizer = Decimal('10') ** -6

def direction(a, b, c):
    val = (b.y - a.y) * (c.x - b.y) - (b.x - a.x)*(c.y - b.y)
    if val == 0:
        return "colinear"
    elif val < 0:
        return "anticlockwise"
    else:
        return "clockwise"


def onlinepointisonsegment(p, l):
    dot_prod = (p.x - l.low.x) * (l.high.x - l.low.x) + (p.y-l.low.y) * (l.high.y - l.low.y)
    if 0 <= dot_prod <= l.length**2:
        return True
    else:
        return False


def isintersect(l1, l2):
    dir1 = direction(l1.high, l1.low, l2.high)
    dir2 = direction(l1.high, l1.low, l2.low)
    dir3 = direction(l2.high, l2.low, l1.high)
    dir4 = direction(l2.high, l2.low, l1.low)

    if dir1 != dir2 and dir3 != dir4:
        return True
    if dir1 is "colinear" and onlinepointisonsegment(l2.high, l1):
        return True
    if dir2 is "colinear" and onlinepointisonsegment(l2.low, l1):
        return True
    if dir3 is "colinear" and onlinepointisonsegment(l1.high, l2):
        return True
    if dir4 is "colinear" and onlinepointisonsegment(l1.low, l2):
        return True
    return False


def is_pointoverlay(p1, p2):
    if p1.x == p2.x and p1.y == p2.y:
        return True
    else:
        return False


class Point:
    def __init__(self, x, y, idex=None):
        self.x = Decimal(x)
        self.y = Decimal(y)
        self.plot = None
        self.idex = idex

    def interior_line(self, seg):
        if abs((self.x - seg.low.x) * (seg.high.y - seg.low.y) - (self.y - seg.low.y) * (seg.high.x - seg.low.x))>1e-6:
            return False
        else:
            if seg.horizontal:
                return (seg.low.x < self.x < seg.high.x) or (seg.high.x < self.x < seg.low.x)
            elif seg.vertical:
                return seg.low.y < self.y < seg.high.y
            else:
                return seg.low.y < self.y < seg.high.y and (
                            seg.low.x < self.x < seg.high.x or seg.high.x < self.x < seg.low.x)

    def draw(self, axis=None, **kwargs):
        if axis is None:
            figure, axis = plt.subplots()
        self.plot = axis.scatter(self.x, self.y, **kwargs)
        axis.text(self.x,self.y, self.__str__())

    def __str__(self):
        return "{}".format(self.idex)

    def __eq__(self, other):
        return self.y == other.y and self.x == other.x

    def __gt__(self, other):
        if self.y > other.y:
            return False
        elif self.y < other.y:
            return True
        elif self.x > other.x:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.y != other.y:
            return self.y > other.y
        else:
            return self.x < other.x

    def __hash__(self):
        return hash(self.x + 100*self.y)


class EndPoint(Point):
    def __init__(self, coord, incident_edge, position):
        super(EndPoint, self).__init__(coord[0], coord[1])
        self.incident_edge = {"upper": [], "lower": []}
        if position:
            self.incident_edge[position].append(incident_edge)

    def merge(self, other):
        if is_pointoverlay(self, other):
            self.incident_edge["upper"] += other.incident_edge["upper"]
            self.incident_edge["lower"] += other.incident_edge["lower"]
            for e_h in other.incident_edge["upper"]:
                e_h.high = self
            for e_l in other.incident_edge["lower"]:
                e_l.low = self
        else:
            raise RuntimeError("The overlaid points must have the same position")


def check_coord(coord):
    p1 = Point(coord[:2])
    p2 = Point(coord[-2:])
    return is_pointoverlay(p1, p2)


class LineSegments(object):
    def __init__(self, coords, idex=None, swline = None):
        """
        :param coords: in the form of xyxy, where the each pair of xy is the coordinates one of the two endpoints.
        """
        endpoint_1 = Point(coords[0].item(), coords[1].item())
        endpoint_2 = Point(coords[2].item(), coords[3].item())

        # if is_pointoverlay(endpoint_1, endpoint_2):
        #     self.high = None
        #     self.low = None
        if endpoint_1 > endpoint_2:
            self.high = endpoint_2
            self.low = endpoint_1
        else:
            self.high = endpoint_1
            self.low = endpoint_2
        # elif endpoint_1.y == endpoint_2.y:
        #     if endpoint_1.x < endpoint_2.x:
        #         self.high = endpoint_1
        #         self.low = endpoint_2
        #     else:
        #         self.high = endpoint_2
        #         self.low = endpoint_1
        # else:
        #     self.high = endpoint_2
        #     self.low = endpoint_1
        self.high = EndPoint([self.high.x, self.high.y], self, "upper")
        self.low = EndPoint([self.low.x, self.low.y], self, "lower")
        self.high.idex = str(idex) + "h"
        self.low.idex = str(idex) + "l"
        self.length = math.sqrt((self.high.x-self.low.x)**2 + (self.high.y-self.low.y)**2)
        self.x = None
        self.plot = None
        # self.sweep_status = None
        if self.high.y == self.low.y:
            self.horizontal = True
        else:
            self.horizontal = False

        if self.high.x == self.low.x:
            self.vertical = True
        else:
            self.vertical = False

        self.swline = swline

        self.params = dict()
        self.get_params()

    def get_params(self):
        self.params["a"] = self.get_a()
        self.params["b"] = self.get_b()
        self.params["hp"] = self.get_hp()

    def get_a(self):
        if self.horizontal:
            return Decimal(0)
        elif self.vertical:
            return np.inf
        else:
            return (self.high.y - self.low.y)/(self.high.x - self.low.x)

    def get_b(self):
        if self.params["a"] != np.inf:
            return self.high.y - self.params["a"] * self.high.x
        else:
            return None

    def get_hp(self):
        if self.params["a"] == 0:
            return np.inf
        else:
            return -1/self.params["a"]

    def sweep(self):
        p = self.swline.current_status
        x, y = p.x, p.y
        if self.low.y == self.high.y and self.low.y == y:
            self.x = p.x
        # elif (self.low.y <= y <= self.high.y) or (self.low.y >= y >= self.high.y):
        #     self.x = self.high.x + (y-self.high.y) * (self.high.x - self.low.x) / (self.high.y - self.low.y)
        else:
            if self.params["a"] == 0:
                self.x = x
            elif self.params["a"] == np.inf:
                self.x = self.high.x
            else:
                self.x = (y - self.params["b"])/self.params["a"]
            # else:
            #     self.x = self.high.x


    def draw(self, axis=None, **kwargs):
        if axis is None:
            figure, axis = plt.subplots()

        self.plot = axis.plot([self.low.x, self.high.x], [self.low.y, self.high.y], **kwargs)
        # axis.scatter([self.low.x, self.high.x], [self.low.y, self.high.y])

    def __str__(self):
        return "{}{}".format(str(self.high), str(self.low))

    def __eq__(self, other):
        return self.high == other.high and self.low == other.low

    def __lt__(self, other):
        self.sweep()
        other.sweep()
        if self == other:
            return False
        if abs(self.x - other.x) < 1e-12:
            return self.params["hp"] < other.params["hp"]
        else:
            return self.x < other.x

    def __gt__(self, other):
        return not self.__lt__(other) and not self == other

    def __hash__(self):
        return hash(int(self.low.x + Decimal(100)*self.low.y + Decimal(1e4)*self.high.x + Decimal(1e6)*self.high.y))


def create_linesegment_from_enpoint_list(point_list, swline=None):
    segments = []
    for i, seg in enumerate(point_list):
        if not check_coord(seg):
            segments.append(LineSegments(seg, idex=i, swline=swline))
    return segments


if __name__ == "__main__":
    import numpy as np
    from bst import Tree
    from event_queue import EventQueue
    # seg_list = [[0, 2, 1, -1], [2, 0, -1, 1]]

    class TestClass:
        def __init__(self):
            self.current_status = 0

    seg_list = np.random.randint(-8, 8, (6, 2))
    seg_list = [seg_list[:, 0], -5*np.ones((6,)), seg_list[:, 1], 5*np.ones((6,))]
    seg_list = np.stack(seg_list, axis=1)

    seg_list = create_linesegment_from_enpoint_list(seg_list, swline=TestClass())


    ep_list = []
    seg_tree = Tree()
    for seg in seg_list:
        seg.sweep()
        seg_tree.insert(seg)
        ep_list += [seg.high, ]
        ep_list += [seg.low, ]
    # heapify(seg_list)

    fig0, ax0 = plt.subplots()
    seg_tree.draw(ax0)
    fig, ax = plt.subplots()
    for seg in seg_list:
        seg.draw(ax)
        seg.high.draw(ax)
        seg.low.draw(ax)
        # seg.high.draw(ax)

    plt.show()

    # seg_list[0].high.draw(ax)
    # seg_list[0].high.incident_edge.draw(ax)

    plt.show()
