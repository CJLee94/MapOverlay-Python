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
        self.U = set()
        self.L = set()
        self.C = set()
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

    def add_to_set(self, seg, S):
        S.add(seg)
        if S is self.L:
            seg.plot[0].set_color("green")
        elif S is self.U:
            seg.plot[0].set_color("red")
        elif S is self.C:
            seg.plot[0].set_color("yellow")

    def clean_sets(self):
        for S in (self.L, self.C, self.U):
            while len(S) > 0:
                e = S.pop()
                if e not in self.t.s:
                    e.plot[0].set_color("blue")

    def handle_eventpoint(self, p):
        # begin the starting phase of handling the point
        p.plot.set_color("green")
        self.ax.set_title("handling point ({}, {})".format(p.x, p.y))
        plt.waitforbuttonpress()

        # Let U be the set of segments whose upper endpoint is p
        self.ax.set_title("all the segments whose upper endpoint is p")
        for e in p.incident_edge["upper"]:
            e.sweep(p.y)
            self.add_to_set(e, self.U)

        # self.set_color(U, "red")
        plt.waitforbuttonpress()
        # self.set_color(U, "blue")

        for seg in self.t.s:
            if seg in p.incident_edge["lower"]:
                self.t.delete(seg)
                self.add_to_set(seg, self.L)
            elif p.interior_line(seg):
                self.t.delete(seg)
                self.add_to_set(seg, self.C)
            else:
                self.delete(seg)

        self.ax.set_title("all the segments whose lower endpoint is p")
        # self.set_color(L, "red")
        plt.waitforbuttonpress()
        # self.set_color(L, "blue")

        self.ax.set_title("all the segments who contains p")
        # self.set_color(C, "red")
        plt.waitforbuttonpress()
        # self.set_color(C, "blue")

        LUC = self.L | self.U | self.C
        if len(LUC) > 1:
            p.intersection = LUC
            self.intersection.add(p)

        # delete the segments in L|C from t
        LC = self.L | self.C
        UC = self.U | self.C
        for e in LC:
            self.t.delete(e)
            # e.plot[0].set_color("blue")

        for e in UC:
            e.sweep(p.y)
            self.t.insert(e)
            # e.plot[0].set_color("black")

        plt.waitforbuttonpress()

        seg_p = LineSegments([p.x, p.y - 1, p.x, p.y + 1])
        seg_p.x = p.x
        if len(UC) == 0:
            s_l = self.t.left_neighbor(seg_p)
            s_r = self.t.right_neighbor(seg_p)
            if s_l is not None and s_r is not None:
                self.find_new_event(s_l, s_r, p)

        else:
            s_prime = min(UC)
            try:
                s_l = self.t.left_neighbor(s_prime)
            except:
                import pdb
                pdb.set_trace()
            if s_l:
                self.find_new_event(s_l, s_prime, p)
            s_pprime = max(UC)
            s_r = self.t.right_neighbor(s_pprime)
            if s_r:
                self.find_new_event(s_pprime, s_r, p)

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