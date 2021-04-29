import bisect


class SortedSet:
    def __init__(self):
        self.s = []

    def insert(self, k):
        k.plot[0].set_color("black")
        bisect.insort(self.s, k)

    def delete(self, k):
        k.plot[0].set_color("blue")
        self.s.remove(k)

    def left_neighbor(self, x):
        i = bisect.bisect_left(self.s, x)
        if i-1 >= 0:
            return self.s[i-1]
        return None

    def right_neighbor(self, x):
        i = bisect.bisect_right(self.s, x)
        if i<len(self.s):
            return self.s[i]
        return None


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from line_segment import create_linesegment_from_enpoint_list

    seg_list = np.random.randint(-5, 5, (6, 4))

    seg_list = create_linesegment_from_enpoint_list(seg_list)

    ep_list = []
    plt.ion()
    fig, ax = plt.subplots()
    for seg in seg_list:
        ep_list += [seg.high, ]
        ep_list += [seg.low, ]
        seg.draw(ax, color="blue")
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    plt.show()

    sweep_line = np.linspace(min(ep_list).y, max(ep_list).y, 10)
    x_s = np.linspace(-5, 5, 10)

    # fig, ax = plt.subplots()

    for sline in sweep_line:
        t = SortedSet()

        for seg in seg_list:
            seg.sweep(sline)
            if seg.x is not None:
                t.insert(seg)

        line_s = ax.plot(x_s, [sline] * len(x_s), "--", color="green")
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        plt.waitforbuttonpress()
        for seg in t.s:
            seg.plot[0].set_color("red")

        # plt.show()
            plt.waitforbuttonpress()

        for seg in seg_list:
            seg.plot[0].set_color("blue")

        line_s.pop(0).remove()

        # ax.clear()
