import bisect

from data_struct.line_segment import LineSegments
from data_struct.event_queue import EventQueue, EPEventQueue
from data_struct.bst import SortedSet, SegTree
from intersection import intersection
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from decimal import *
# %matplotlib notebook
import numpy as np
# plt.ion()
quantizer = Decimal('10') ** -6


class SweepLine:
    def __init__(self, segments, fig=None, axes=None, draw=True, verbose=False):
        self.draw = draw
        self.q = EPEventQueue()
        # self.t = SortedSet()
        self.t = SegTree()
        self.intersection = set()
        if draw:
            # plt.ion()
            self.fig = fig
            self.ax, self.ax_2 = axes
            # self.fig, self.ax_2 = plt.subplots()
            # self.ax.set_aspect(1)
            self.ax.set_axis_off()
            # self.ax_2.set_axis_off()
            self.seg_a_c = mcolors.CSS4_COLORS["royalblue"]
            self.seg_i_c = mcolors.CSS4_COLORS["lightsteelblue"]
            self.pt_a_c = mcolors.CSS4_COLORS["red"]
            self.ani_count = 0
        else:
            self.fig, self.ax = None, None
        self.seg_list = []
        self.U = set()
        self.L = set()
        self.C = set()
        self.h = []
        # self.C = []
        for idex, seg in enumerate(segments):
            seg = LineSegments(seg, idex=idex, swline=self)
            self.seg_list.append(seg)
            if draw:
                seg.draw(self.ax, color=self.seg_i_c)
            # seg.high.draw(self.ax, color="blue")
            # seg.low.draw(self.ax, color="blue")
            self.q.insertKey(seg.high, ax=self.ax)
            self.q.insertKey(seg.low, ax=self.ax)
        self.pause = self.pause_draw if draw else lambda *a: None
        if draw:
            self.x_min = segments[:, [0, 2]].min()
            self.x_max = segments[:, [0, 2]].max()
            self.y_max = self.q.getMin().y
            self.y_min = Decimal(self.ax.get_ylim()[0])
            self.sline = None
            self.recback = None
            self.initilize_sweepline()
        self.current_status = self.q.getMin()
        if verbose:
            self.print_ = print
        else:
            self.print_ = lambda *a: None

    def initilize_sweepline(self):
        # y = self.q.getMin().y
        self.sline = self.ax.plot([self.x_min - 0.2, self.x_max + 0.2], [self.y_max, self.y_max], "--", color="gray")
        self.recback = self.ax.axhspan(self.y_min, self.y_max, facecolor='y', alpha=0.25)

    def draw_sweepline(self, y):
        self.sline[0].set_ydata([y, y])
        xy = self.recback.get_xy()
        xy[1:3, -1] = y
        self.recback.set_xy(xy)


    def move_line(self, y1, y2, time_interval = 0.005):
        # y_min = self.ax.get_ylim[0]
        scales_number = int((y1-y2)/(self.y_max-self.y_min)*100)
        scales = np.linspace(float(y2), float(y1), scales_number)
        for s in scales[::-1]:
            self.draw_sweepline(s)
            # t = 0
            # plt.waitforbuttonpress()
            # self.pause
            # self.fig.canvas.draw_idle()
            plt.pause(time_interval)

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
            if self.draw:
                self.move_line(self.current_status.y, p.y)
                self.draw_sweepline(p.y)
                self.current_status = p
            self.handle_eventpoint(p)

    def add_to_sortedlist(self, lst, elem):
        bisect.insort_left(lst, elem)

    def add_to_t(self, seg):
        self.add_to_sortedlist(self.t.s, seg)

    def add_to_UC(self, seg):
        if seg.horizontal:
            self.h.append(seg)
        else:
            bisect.insort_left(self.UC, seg)

    def clean_sets(self):
        self.L = set()
        self.U = set()
        self.C = set()

    def check(self, l):
        if any(element.x is None for element in l):
            import pdb
            pdb.set_trace()

    def pause_draw(self):
        plt.waitforbuttonpress()
        self.fig.savefig("../animation/anim_{}.png".format(self.ani_count))
        self.ani_count += 1

    def inactivate(self, seg):
        seg.plot[0].set_color(self.seg_i_c)
        if seg.low not in self.intersection:
            seg.low.plot.set_color(self.seg_i_c)
        if seg.high not in self.intersection:
            seg.high.plot.set_color(self.seg_i_c)

    def handle_eventpoint(self, p):
        # begin the starting phase of handling the point
        if self.draw:
            p.plot.set_color("red")
            self.ax.set_title("handling point ({}, {})".format(p.x, p.y))
        # self.pause()
        plt.pause(0.1)
        # Let U be the set of segments whose upper endpoint is p
        # self.ax.set_title("all the segments whose upper endpoint is p")
        self.print_("find line segments whose upper endpoint is current point")
        for seg in p.incident_edge["upper"]:
            seg.sweep()
            self.U.add(seg)
            self.print_("line segment:(({}, {}), ({}, {}))".format(seg.high.x, seg.high.y, seg.low.x, seg.low.y))

        t_s = self.t.node_list.copy()
        for seg in t_s:
            # seg = self.t.s.pop()
            self.print_("processing segment (({}, {}), ({}, {}))".format(seg.high.x, seg.high.y, seg.low.x, seg.low.y))
            if seg in p.incident_edge["lower"]:
                self.t.remove(seg)
                self.L.add(seg)
                if self.draw:
                    self.inactivate(seg)
                self.print_("its lower endpoint is p")
            elif p.interior_line(seg):
                self.t.remove(seg)
                self.C.add(seg)
                if self.draw:
                    self.inactivate(seg)
                self.print_("p is contained in the segment")
            else:
                self.print_("segment is not related to p")
                # self.add_to_sortedlist(t_s, seg)
                if self.draw:
                    seg.plot[0].set_color(self.seg_a_c)
                    seg.low.plot.set_color(self.seg_a_c)
                    seg.high.plot.set_color(self.seg_a_c)
            # seg.sweep()
        t_s = self.t.node_list.copy()
        for seg in t_s:
            seg.sweep()

        LUC = self.L | self.U | self.C
        self.print_("The number of segments related to this point is {}".format(len(LUC)))
        if len(LUC) > 1:
            p.intersection = LUC
            self.intersection.add(p)

        UC = self.U | self.C
        for seg_i, seg in enumerate(UC):
            self.t.insert(seg)

            if self.draw:
                if seg.low.plot is None or seg.high.plot is None:
                    import pdb
                    pdb.set_trace()
                seg.plot[0].set_color(self.seg_a_c)
                seg.low.plot.set_color(self.seg_a_c)
                seg.high.plot.set_color(self.seg_a_c)
            # e.plot[0].set_color("black")

        self.print_("Active Segments:")
        for seg in self.t.node_list:
            self.print_("({}, {}), ({}, {})".format(seg.high.x, seg.high.y, seg.low.x, seg.low.y))
        # plt.waitforbuttonpress()
        if self.draw:
            self.t.draw(self.ax_2)
        # self.pause()
        if len(UC) == 0:
            s_l = self.t.leftNeighbor(p)
            s_r = self.t.rightNeighbor(p)
            if s_l is not None and s_r is not None:
                self.find_new_event(s_l.val, s_r.val, p)

        else:
            s_prime = min(UC)
            s_l_node = self.t.leftNeighbor(s_prime)

            if s_l_node:
                self.find_new_event(s_l_node.val, s_prime, p)
            s_pprime = max(UC)
            s_r_node = self.t.rightNeighbor(s_pprime)

            if s_r_node:
                self.find_new_event(s_pprime, s_r_node.val, p)

        # self.pause()
        if self.draw:
            if len(UC) > 0 or p in self.intersection:
                p.plot.set_color(self.seg_a_c)
            else:
                p.plot.set_color(self.seg_i_c)

        self.pause()
        self.clean_sets()

    def find_new_event(self, s_l, s_r, p):
        c1 = intersection(s_l, s_r)
        if c1 is not None:
            if c1.y < p.y or (c1.y == p.y and c1.x > p.x) and c1 not in self.q.heap:
                # c1.draw(self.ax, color="blue")
                c1.idex = len(self.intersection)+1
                self.print_("find new intersection ({}, {})".format(c1.x, c1.y))
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
    import time
    import os
    from scipy.optimize import curve_fit
    from matplotlib.widgets import Button
    from data_struct.line_segment import create_linesegment_from_enpoint_list

    import matplotlib.pyplot as plt

    class Index:
        def __init__(self, fig=None):
            self.seg_list = []
            if fig is None:
                self.fig = plt.figure()
            else:
                self.fig = fig
            self.ax, self.ax2 = fig.subplots(1, 2)
            self.ax2.set_axis_off()
            self.ax.set_xlim([-11, 11])
            self.ax.set_ylim([-11, 11])
            plt.subplots_adjust(bottom=0.2)
            self.sl = None

        def draw(self, event):

            for i in range(10):
                xy = plt.ginput(2)
                x = [p[0] for p in xy]
                y = [p[1] for p in xy]
                line = self.ax.plot(x, y)
                self.ax.figure.canvas.draw()

                # self.lines.append(line)
                self.seg_list.append([xy[0][0], xy[0][1], xy[1][0], xy[1][1]])
            print(self.seg_list[:9])

        def random_generate(self, event):
            self.seg_list = np.random.randint(-10, 10, (10, 4))
            for coords in self.seg_list:
                line = self.ax.plot(coords[[0, 2]], coords[[1, 3]])
            self.ax.figure.canvas.draw()

        def preset1(self, event):
            self.seg_list = np.array([[-10.329024609442223, 5.700463246219737, -7.293439394229113, -0.332663228738749],
                                      [-2.3328489205881766, 4.065990735075605, -8.033826032085969, -2.6161174722489307],
                                      [-1.5554429508384793, 7.575299361943884, 2.442644893588543, -1.750808495760861],
                                      [3.0349542038740296, 3.561227165457563, -2.7030422395166056, -3.265099204614982],
                                      [8.735931315371822, 4.931299711563675, 5.5892881044801825, -1.1739358447688133]])
            for coords in self.seg_list:
                line = self.ax.plot(coords[[0, 2]], coords[[1, 3]])
            self.ax.figure.canvas.draw()

        def preset2(self, event):
            self.seg_list = np.array([[-9.329502648335467, 4.114063455991609, -7.182381398550584, -1.8709902980508701],
                                      [8.84698931105035, 2.0469364566034436, 6.477752069908409, -0.5249541124027637],
                                      [-6.960265407193528, 2.3834455030154693, -1.868640824382341936124537715, 1.712549028530946292255423991],
                                      [-6.404975428800885, 5.171663316143693, -1.868640824382341936124537715, 1.712549028530946292255423991],
                                      [-3.9987188557661035, 6.565772222707803, 0.07340765244660474, -2.712262914080938],
                                      [-6.182859437443829, -0.8854995192727912, 3.5162055184809873, 4.955336072021677],
                                      [-1.868640824382341936124537715, 1.712549028530946292255423991, -3.1472742222307186, -3.0247356000349637]])
            for coords in self.seg_list:
                line = self.ax.plot(coords[[0, 2]], coords[[1, 3]])
            self.ax.figure.canvas.draw()

        def sweep(self, event):
            if len(self.seg_list)!=0:
                self.sl = SweepLine(np.array(self.seg_list), fig=self.fig, axes=(self.ax, self.ax2), draw=True, verbose=True)
                self.sl.find_intersection()

        def clear(self, event):
            self.seg_list = []
            # self.ax, self.ax2 = fig.subplots(1, 2)
            self.ax.clear()
            self.ax2.clear()
            self.ax2.set_axis_off()
            self.ax.set_xlim([-11, 11])
            self.ax.set_ylim([-11, 11])
            plt.subplots_adjust(bottom=0.2)
            self.sl = None

    fig = plt.figure()
    callback = Index(fig=fig)
    axpre1 = plt.axes([0.26, 0.05, 0.1, 0.075])
    bpre1 = Button(axpre1, "Preset1")
    bpre1.on_clicked(callback.preset1)
    axpre2 = plt.axes([0.37, 0.05, 0.1, 0.075])
    bpre2 = Button(axpre2, "Preset2")
    bpre2.on_clicked(callback.preset2)
    axdraw = plt.axes([0.48, 0.05, 0.1, 0.075])
    bdraw = Button(axdraw, "Draw")
    bdraw.on_clicked(callback.draw)
    axrandom = plt.axes([0.59, 0.05, 0.1, 0.075])
    brandom = Button(axrandom, "Random")
    brandom.on_clicked(callback.random_generate)
    axsweep = plt.axes([0.70, 0.05, 0.1, 0.075])
    bsweep = Button(axsweep, "Sweep")
    bsweep.on_clicked(callback.sweep)
    axclear = plt.axes([0.81, 0.05, 0.1, 0.075])
    bclear = Button(axclear, "Clear")
    bclear.on_clicked(callback.clear)

    plt.show()

    number_size = np.arange(3, 50)
    trial_per_size = 10
    experiment_results = np.zeros((2, trial_per_size, len(number_size)))
    if not os.path.exists("experiment_result.npy"):

        for sindex, ssize in enumerate(number_size):
            for trial in range(trial_per_size):
                seg_list = np.random.randint(-10, 10, (ssize, 4))
                # seg_list = np.array([[0, 2, -3, 4],
                #                      [-3, -2, -3, -5],
                #                      [-1, -3, -3, -2],
                #                      [3, -1, 2, 4],
                #                      [-5, -5, -2, 3],
                #                      [-5, 3, 4, 3]])
                sl = SweepLine(seg_list, draw=False)
                tic = time.time()
                sl.find_intersection()
                toc = time.time()

                # sl.ax.set_title("all the intersection points")
                # for points in sl.intersection:
                #     points.draw(sl.ax, color="pink")
                print("The number of intersection points:{}".format(len(sl.intersection)))
                experiment_results[0, trial, sindex] = toc - tic
                experiment_results[1, trial, sindex] = len(sl.intersection)
                # plt.waitforbuttonpress()
                sl.pause()
        np.save("experiment_result.npy", experiment_results)
    else:
        experiment_results = np.load("experiment_result.npy")


    def log_function(x, a):
        return a * (x[0] + x[1]) * np.log(x[0])

    def log_function_upper(x, a):
        return a * (x + (x-1)*x/2) * np.log(x[0])
    def log_function_lower(x, a):
        return a * x * np.log(x[0])
    x = np.stack([number_size]*trial_per_size, axis=0).flatten()
    x = np.stack([x, experiment_results[1].flatten()], axis=0)
    y = experiment_results[0].flatten()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    param, param_cov = curve_fit(log_function, x, y)

    number_size = np.repeat(number_size[None], trial_per_size, axis=0)

    ax.scatter(number_size.flatten(), experiment_results[1].flatten(), experiment_results[0].flatten())

    fig2, ax2 = plt.subplots()
    ax2.scatter(number_size.flatten(), experiment_results[0].flatten(),
                color=mcolors.CSS4_COLORS["lightsteelblue"])
    # ax2.plot(number_size[0], experiment_results[0].mean(axis=0), color="red")
    ax2.plot(number_size[0], log_function_upper(number_size[0], np.array([param.item()]*len(number_size[0]))),
             color="red", label="Upper bound")
    ax2.plot(number_size[0], log_function_lower(number_size[0], np.array([param.item()] * len(number_size[0]))),
             color="green", label="Lower bound")
    ax2.plot(number_size[0], experiment_results[0].mean(axis=0),
             color="orange", label="Mean for each input length")
    ax2.legend()
    ax2.set_title("Sweep Line Performance")
    ax2.set_xlabel("Input length")
    ax2.set_ylabel("Running time")
    fig3, ax3 = plt.subplots()
    ax3.scatter(experiment_results[1].flatten(), experiment_results[0].flatten())

    plt.show()
