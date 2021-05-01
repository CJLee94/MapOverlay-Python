import bisect

from data_struct.line_segment import LineSegments
from data_struct.event_queue import EventQueue, EPEventQueue
from data_struct.bst import SortedSet
from intersection import intersection
import matplotlib.pyplot as plt
import numpy as np



class SweepLine:
    def __init__(self, segments):
        self.q = EPEventQueue()
        self.t = SortedSet()
        self.intersection = set()
        # self.delta = 1e-5
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.seg_list = []
        self.UC = []
        self.LC = []
        self.h = []
        # self.C = []
        for seg in segments:
            seg = LineSegments(seg)
            self.seg_list.append(seg)
            seg.draw(self.ax, color="blue")
            # seg.high.draw(self.ax, color="blue")
            # seg.low.draw(self.ax, color="blue")
            self.q.insertKey(seg.high, ax=self.ax)
            self.q.insertKey(seg.low, ax=self.ax)

    def find_intersection(self):
        """
        find the intersection of segments in the input segment set s
        :param s: A set S of line segments in the plane
        :return: The set of intersection points among the segments in S, with for each intersection point the segments that
        contain it.
        """

        # Initialize an empty event queue Q. Next, insert the segment endpoints into Q; when an upper endpoint is
        #  inserted, the corresponding segment should be stored with it.

        # TODO: Sweep Line
        while len(self.q.heap) != 0:
            p = self.q.extractMin()
            self.handle_eventpoint(p)

    def add_to_t(self, seg):
        bisect.insort_left(self.t.s, seg)

    def add_to_UC(self, seg):
        if seg.horizontal:
            self.h.append(seg)
        else:
            bisect.insort_left(self.UC, seg)
        # S.add(seg)
        # if S is self.L:
        #     seg.plot[0].set_color("green")
        # elif S is self.U:
        #     seg.plot[0].set_color("red")
        # elif S is self.C:
        #     seg.plot[0].set_color("yellow")

    def clean_sets(self):
        self.LC = []
        self.UC = []
        self.h = []

    def check(self, l):
        if any(element.x is None for element in l):
            import pdb
            pdb.set_trace()

    def handle_eventpoint(self, p):
        # begin the starting phase of handling the point
        p.plot.set_color("green")
        self.ax.set_title("handling point ({}, {})".format(p.x, p.y))
        plt.waitforbuttonpress()

        # Let U be the set of segments whose upper endpoint is p
        # self.ax.set_title("all the segments whose upper endpoint is p")
        print("find line segments whose upper endpoint is current point")
        for seg in p.incident_edge["upper"]:
            if seg.horizontal:
                seg.sweep(p.y)
                seg.x = p.x
            else:
                seg.sweep(p.y-1e-3)
            self.add_to_UC(seg)
            print("line segment:(({}, {}), ({}, {}))".format(seg.high.x, seg.high.y, seg.low.x, seg.low.y))
            self.check(self.UC)
            # if seg.x == 1e12:
            #     self.UC.append(seg)
            # else:
            #     bisect.insort_left(self.UC, seg)
            # self.add_to_set(e, self.U)

        # self.set_color(U, "red")
        # plt.waitforbuttonpress()
        # self.set_color(U, "blue")

        t_s = self.t.s.copy()
        for seg in t_s:
            print("processing segment (({}, {}), ({}, {}))".format(seg.high.x, seg.high.y, seg.low.x, seg.low.y))
            if any(elem is seg for elem in p.incident_edge["lower"]):
                self.t.delete(seg)
                # self.add_to_set(seg, self.L)
                self.LC.append(seg)
                seg.plot[0].set_color("blue")
                print("its lower endpoint is p")
            elif p.interior_line(seg):
                self.t.delete(seg)
                if seg.horizontal:
                    seg.sweep(p.y)
                    seg.x = p.x
                else:
                    seg.sweep(p.y-1e-3)
                self.add_to_UC(seg)
                self.check(self.UC)
                # bisect.insort_left(self.UC, seg)
                self.LC.append(seg)
                seg.plot[0].set_color("blue")
                # self.t.insert(seg)
                # self.add_to_set(seg, self.C)
                print("p is contained in the segment")
            else:
                self.t.delete(seg)
                if seg.horizontal:
                    seg.sweep(p.y)
                    seg.x = p.x
                else:
                    seg.sweep(p.y)
                print("segment is not related to p")
                if seg not in self.t.s:
                    self.add_to_t(seg)
                    seg.plot[0].set_color("black")

        # self.ax.set_title("all the segments whose lower endpoint is p")
        # # self.set_color(L, "red")
        # plt.waitforbuttonpress()
        # # self.set_color(L, "blue")
        #
        # self.ax.set_title("all the segments who contains p")
        # # self.set_color(C, "red")
        # plt.waitforbuttonpress()
        # # self.set_color(C, "blue")

        LUC = set(self.LC) | set(self.UC)
        if len(LUC) > 1:
            p.intersection = LUC
            self.intersection.add(p)

        # delete the segments in L|C from t
        # LC = self.L | self.C
        # UC = self.U | self.C
        # for e in self.LC:
        #     self.t.delete(e)
            # e.plot[0].set_color("blue")

        seg_p = LineSegments([p.x, p.y - 1, p.x, p.y + 1])
        seg_p.x = p.x
        self.UC += self.h
        insert_idx = bisect.bisect_right(self.t.s, seg_p)
        for seg_i, seg in enumerate(self.UC):
            # e.sweep(p.y)
            if seg not in self.t.s:
                if seg_i == 0:
                # self.check(self.t.s)
                    self.t.s.insert(insert_idx, seg)
                else:
                    self.t.s.insert(insert_idx+1, seg)
                seg.plot[0].set_color("black")
            # e.plot[0].set_color("black")

        print("Active Segments:")
        for seg in self.t.s:
            print("({}, {}), ({}, {})".format(seg.high.x, seg.high.y, seg.low.x, seg.low.y))
        plt.waitforbuttonpress()
        if len(self.UC) == 0:
            s_l = self.t.left_neighbor(seg_p)
            s_r = self.t.right_neighbor(seg_p)
            if s_l is not None and s_r is not None:
                self.find_new_event(s_l, s_r, p)

        else:
            s_prime = self.UC[0]
            if self.t.s.index(s_prime) - 1 >= 0:
                s_l = self.t.s[self.t.s.index(s_prime) - 1]
            else:
                s_l = None
            if s_l:
                self.find_new_event(s_l, s_prime, p)
            s_pprime = self.UC[-1]
            if self.t.s.index(s_pprime) + 1 < len(self.t.s):
                s_r = self.t.s[self.t.s.index(s_pprime)+1]
            else:
                s_r = None
            if s_r:
                self.find_new_event(s_pprime, s_r, p)

        plt.waitforbuttonpress()
        self.clean_sets()
        p.plot.set_color("gray")

    def find_new_event(self, s_l, s_r, p):
        c1 = intersection(s_l, s_r)
        if c1 is not None:
            if c1.y < p.y or (c1.y == p.y and c1.x > p.x) and c1 not in self.q.heap:
                # c1.draw(self.ax, color="blue")
                self.intersection.add(c1)
                self.q.insertKey(c1, ax=self.ax)

    # def set_color(self, S, c):
    #     for e in S:
    #         e.plot[0].set_color(c)

    def reset_plot(self):
        self.set_color(self.seg_list, "blue")

        # if s_l.intersect(s_r) or :
if __name__ == "__main__":
    import numpy as np
    from data_struct.line_segment import create_linesegment_from_enpoint_list
    seg_list = np.random.randint(-5, 5, (6, 4))

    sl = SweepLine(seg_list)

    sl.find_intersection()

    sl.ax.set_title("all the intersection points")
    for points in sl.intersection:
        points.draw(sl.ax, color="pink")
    plt.waitforbuttonpress()