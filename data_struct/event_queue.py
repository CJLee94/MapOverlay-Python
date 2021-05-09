from heapq import heappush, heappop, heapify
import itertools
import matplotlib.colors as mcolors

class EventQueue:
    def __init__(self):
        self.heap = []

    def insertKey(self, k):
        heappush(self.heap, k)

    def insertKey(self, k):
        heappush(self.heap, k)

    def extractMin(self):
        return heappop(self.heap)

    def getMin(self):
        return self.heap[0]


class EPEventQueue(EventQueue):
    def __init__(self):
        super(EPEventQueue, self).__init__()

    def insertKey(self, k, ax=None):
        if k not in self.heap:
            if k.plot is not None:
                k.plot.set_color(mcolors.CSS4_COLORS["lightsteelblue"])
            elif ax is not None:
                k.draw(ax, color=mcolors.CSS4_COLORS["lightsteelblue"])
            heappush(self.heap, k)
        else:
            i = self.heap.index(k)
            self.heap[i].merge(k)

    def decreaseKey(self, i, new_val):
        self.heap[i].x = new_val[0]
        self.heap[i].y = new_val[1]

        while i != 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = (self.heap[self.parent(i)], self.heap[i])
            i = self.parent(i)


if __name__ == "__main__":
    import numpy as np
    from line_segment import create_linesegment_from_enpoint_list
    seg_list = np.random.randint(-5, 5, (6, 4))

    seg_list = create_linesegment_from_enpoint_list(seg_list)

    ep_list = EPEventQueue()
    for seg in seg_list:
        ep_list.insertKey(seg.high)
        ep_list.insertKey(seg.low)
    while len(ep_list.heap) != 0:
        ep = ep_list.extractMin()
        print("({},{})".format(ep.x, ep.y))

    pass


