from data_struct.line_segment import EndPoint


def cal_params(l):
    """
    Calculate the parameters of a line segments so that the line segment lies on the line ax + by = c
    :param l: line segment
    :return:
    """

    a = l.high.y - l.low.y
    b = - (l.high.x - l.low.x)
    c = l.low.x * l.high.y - l.high.x * l.low.y
    return a, b, c


def is_on(coord, l):
    return l.low.y <= coord[1] <= l.high.y and (l.low.x <= coord[0] <= l.high.x or l.high.x <= coord[0] <= l.low.x)


def check(l1, l2, coord):
    if None in coord:
        return False
    else:
        return is_on(coord, l1) and is_on(coord, l2)


def intersection(l1, l2):
    """
    Calculate the intersection point of the two line segments
    :param l1: line segment
    :param l2: line segment
    :return:
    """
    a1, b1, c1 = cal_params(l1)
    a2, b2, c2 = cal_params(l2)

    if (a1 * b2 - a2 * b1) == 0:
        return None
    else:
        x_in = (c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1)
        y_in = (c1 * a2 - c2 * a1) / (b1 * a2 - b2 * a1)

    if check(l1, l2, (x_in, y_in)):
        return EndPoint((x_in, y_in), None, None)
    else:
        return None


if __name__ == "__main__":
    from data_struct.line_segment import LineSegments
    import matplotlib.pyplot as plt
    import numpy as np

    l1_coords = np.random.randint(-5, 5, (4,))
    l2_coords = np.random.randint(-5, 5, (4,))
    l1 = LineSegments(l1_coords)
    l2 = LineSegments(l2_coords)

    print(l1_coords)
    print(l2_coords)

    fig, ax = plt.subplots()
    l1.draw(ax)
    l2.draw(ax)

    p = intersection(l1, l2)

    if p is not None:
        plt.scatter(p.x, p.y)
    else:
        print("No intersect points")

    plt.show()
