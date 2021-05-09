import bisect

import matplotlib.pyplot as plt


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

class TreeNode(object):
    def __init__(self, x, parent=None):
        self.val = x
        self.left = None
        self.right = None
        self.parent = parent
        self.height = 0
        self.bf = 0
        self.level = 0


class Tree:
    def __init__(self):
        self.root = None
        self.node_count = 0

    def insert(self, k):
        if k is None:
            return False
        if not self.contains(k):
            self.root = self._insert(self.root, k)
            self.node_count += 1
            return True
        return False

    def _insert(self, node, k):
        if node is None:
            return TreeNode(k)

        if k < node.val:
            node.left = self._insert(node.left, k)
            node.left.parent = node
        else:
            node.right = self._insert(node.right, k)
            node.right.parent = node

        self.update(node)

        return self.balance(node)

    def update(self, node):
        lh = -1
        rh = -1
        # level = 0
        if node.left is not None:
            lh = node.left.height
        if node.right is not None:
            rh = node.right.height
        node.height = 1 + max(lh, rh)

        node.bf = rh - lh

    def balance(self, node):
        if node.bf == -2:
            pass
            if node.left.bf <= 0:
                return self.leftLeftCase(node)
            else:
                return self.leftRightCase(node)
        elif node.bf == 2:
            pass
            if node.right.bf >= 0:
                return self.rightRightCase(node)
            else:
                return self.rightLeftCase(node)
        return node

    def leftLeftCase(self, node):
        return self.rightRotate(node)

    def leftRightCase(self, node):
        node.left = self.leftRotate(node.left)
        return self.leftLeftCase(node)

    def rightRightCase(self, node):
        return self.leftRotate(node)

    def rightLeftCase(self, node):
        node.right = self.rightRotate(node.right)
        return self.rightRightCase(node)

    def leftRotate(self, node):
        P = node.parent
        B = node.right
        node.right = B.left
        if B.left is not None:
            B.left.parent = node
        B.left = node
        node.parent = B
        B.parent = P

        if P is not None:
            if P.left is node:
                P.left = B
            else:
                P.right = B

        self.update(node)
        self.update(B)
        return B

    def rightRotate(self, node):
        P = node.parent
        B = node.left
        node.left = B.right
        if B.right is not None:
            B.right.parent = node
        B.right = node
        node.parent = B
        B.parent = P

        if P is not None:
            if P.left is node:
                P.left = B
            else:
                P.right = B

        self.update(node)
        self.update(B)
        return B

    def contains(self, k):
        return self._contains(self.root, k)

    def _contains(self, node, k):
        if node is None:
            return False
        if node.val == k:
            return True
        elif k < node.val:
            return self._contains(node.left, k)
        else:
            return self._contains(node.right, k)

    def minValueNode(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def maxValueNode(self, node):
        current = node
        while current.right is not None:
            current = current.right
        return current

    def remove(self, k):
        if k is None:
            return False
        if self.contains(k):
            self.root = self._remove(self.root, k)
            self.node_count -= 1
            return True
        return False

    def _remove(self, node, k):
        if node is None:
            return node

        if node.val == k:
            if node.left is None:

                return node.right
            elif node.right is None:
                return node.left
            else:
                if node.left.height > node.right.height:
                    successorValue = self.maxValueNode(node.left).val
                    node.val = successorValue
                    node.left = self._remove(node.left, successorValue)
                else:
                    successorValue = self.minValueNode(node.right).val
                    node.val = successorValue
                    node.right = self._remove(node.right, successorValue)
            # temp = self.minValueNode(node.right)
            #
            # node.val = temp.val
            #
            # node.right = self.remove(node.right, temp.val)
        elif k < node.val:
            node.left = self._remove(node.left, k)
        else:
            node.right = self._remove(node.right, k)

        self.update(node)
        return self.balance(node)
        # return node
        # self.update(node)
        # return self.balance(node)

    def traverse(self):
        return self._traverse(self.root)

    def _traverse(self, node):
        stack = []
        if node is not None:
            stack += self._traverse(node.left)
            stack.append(node.val)
            stack += self._traverse(node.right)
        return stack

    def draw(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlim(-2**(self.root.height-1)-0.5, 2**(self.root.height-1)+0.5)
        ax.set_ylim(-2*(self.root.height)-0.5, 0.5)
        ax.set_aspect(1)
        self._draw(self.root, 0, 0, ax)

    def _draw(self, node, par_x, par_y, ax):
        self.update_level()
        if node is not None:
            if node.parent is None:
                x = 0
                y = 0
            else:
                dist = 2**(self.root.height - node.level)
                ratio = 0.25 / np.sqrt((0.5*dist)**2+2**2)
                y = par_y - 2
                if node is node.parent.left:
                    x = par_x - 0.5*dist
                    ax.plot([x+0.5*dist*ratio, par_x-0.5*dist*ratio], [y+2*ratio, par_y-2*ratio], color="black")
                elif node is node.parent.right:
                    x = par_x + 0.5*dist
                    ax.plot([x - 0.5 * dist * ratio, par_x + 0.5 * dist * ratio],
                            [y + 2 * ratio, par_y - 2 * ratio], color="black")

            ax.add_artist(plt.Circle((x, y), radius=.25))
            ax.text(x, y, str(node.val), ha="center", va="center", color="w", fontsize=10)
            self._draw(node.left, x, y, ax)
            self._draw(node.right, x, y, ax)

    def update_level(self):
        self._update_level(self.root)

    def _update_level(self, node):
        if node is not None:
            node.level = 1+node.parent.level if node.parent is not None else 0
            self._update_level(node.left)
            self._update_level(node.right)

    def print_tree(self, node):
        self.update_level()
        if node is not None:
            print("\t"*(node.level)+str(node.val)+"({})".format(node.level))
            self.print_tree(node.right)
            self.print_tree(node.left)


if __name__ == "__main__":
    import numpy as np

    number_sample = np.random.randint(0, 100, 16)
    # number_sample = np.array([74, 78,50, 97, 57, 25, 39, 21])
    test = Tree()
    for i in number_sample:
        test.insert(i)
        # test.draw()
        # test.print_tree(test.root)
    test.remove(number_sample[1])
    print("remove {}".format(number_sample[1]))
    test.draw()
    plt.waitforbuttonpress()
    # test.print_tree(test.root)
    # test
    # import matplotlib.pyplot as plt
    # from line_segment import create_linesegment_from_enpoint_list
    #
    # seg_list = np.random.randint(-5, 5, (6, 4))
    #
    # seg_list = create_linesegment_from_enpoint_list(seg_list)
    #
    # ep_list = []
    # plt.ion()
    # fig, ax = plt.subplots()
    # for seg in seg_list:
    #     ep_list += [seg.high, ]
    #     ep_list += [seg.low, ]
    #     seg.draw(ax, color="blue")
    # ax.set_xlim([-6, 6])
    # ax.set_ylim([-6, 6])
    # plt.show()
    #
    # sweep_line = np.linspace(min(ep_list).y, max(ep_list).y, 10)
    # x_s = np.linspace(-5, 5, 10)
    #
    # # fig, ax = plt.subplots()
    #
    # for sline in sweep_line:
    #     t = SortedSet()
    #
    #     for seg in seg_list:
    #         seg.sweep(sline)
    #         if seg.x is not None:
    #             t.insert(seg)
    #
    #     line_s = ax.plot(x_s, [sline] * len(x_s), "--", color="green")
    #     ax.set_xlim([-6, 6])
    #     ax.set_ylim([-6, 6])
    #     plt.waitforbuttonpress()
    #     for seg in t.s:
    #         seg.plot[0].set_color("red")
    #
    #     # plt.show()
    #         plt.waitforbuttonpress()
    #
    #     for seg in seg_list:
    #         seg.plot[0].set_color("blue")
    #
    #     line_s.pop(0).remove()

        # ax.clear()
