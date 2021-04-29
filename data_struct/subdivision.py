class Vertex:
    def __init__(self, coord, incident_edge=None):
        """
        The vertex in a subdivision
        :param coord: The coordinates of the vertex
        :param incident_edge: The incident edge of the vertex, should be in an edge structure
        """
        self.coord = coord
        self.incident_edge = incident_edge


class Face:
    def __init__(self, outer_c=None, inner_c=None):
        self.outer_c = outer_c
        self.inner_c = inner_c


class HalfEdge:
    def __init__(self, origin=None, twin=None, inc_f=None, nxt=None, prev=None):
        self.origin = origin
        self.twin = twin
        self.inc_f = inc_f
        self.nxt = nxt
        self.prev = prev

class VertexList:
    def __init__(self):
        self.vertices = []

    def get_coords(self):
        coord_list = []
        for v in self.vertices:
            coord_list.append(v.coord)
        return coord_list

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

class PlanarSubdivision:
    def __init__(self):
        self.v_list = VertexList()
        self.f_list = FaceList()
        self.he_list = HalfEdgeList()
        self.current_v = None

    def add_start_vertex(self, coord):
        if coord not in self.v_list.get_coords():
            self.current_v = Vertex(coord)
            self.v_list.add_vertex(self.current_v)

    def add_end_vertex(self, coord):
        # TODO: Check if the new edge intersect with others
        if self.current_v.incident_edge is None:
            self.current_v.incident_edge =