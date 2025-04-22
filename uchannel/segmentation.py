import gmsh
import numpy as np
import networkx as nx
from collections import defaultdict


class FaceAdjacencyGraph:
    """ Class for building a face-adjacency graph using gmsh and networkx """
    def __init__(self, dim: int = 2) -> None:
        """
        Initialize the FaceAdjacencyGraph class using Gmsh.
        :param dim: The dimension of the faces to consider (2 for surfaces).
        """
        self.graph = nx.MultiDiGraph()
        self.dim = dim  # Face dimension, usually 2 for surfaces

    def create_graph(self) -> nx.MultiDiGraph:
        """
        Build the adjacency graph from the current Gmsh model.
        :return: A MultiDiGraph of face adjacencies.
        """
        self._add_faces()
        self._add_edges()
        return self.graph

    def _add_faces(self) -> None:
        """
        Add all 2D surfaces as nodes in the graph, including their parametric center (center of mass).
        """
        surfaces = gmsh.model.getEntities(dim=self.dim)

        for dim, tag in surfaces:
            center = gmsh.model.occ.getCenterOfMass(dim, tag)
            self.graph.add_node(tag, center=tuple(center))

    def _add_edges(self) -> None:
        """
        Connect surfaces that share edges.
        """
        edge_to_faces = defaultdict(set)

        # Map which surfaces share which edges
        for dim, tag in gmsh.model.getEntities(dim=self.dim):
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, recursive=False)
            for edge_dim, edge_tag in bnd:
                if edge_dim == 1:  # Only edges
                    edge_to_faces[edge_tag].add(tag)

        # Add edges between surfaces that share the same edge
        for edge_tag, faces in edge_to_faces.items():
            faces = list(faces)
            if len(faces) == 2:
                a, b = faces
                if not self.graph.has_edge(a, b):
                    self.graph.add_edge(a, b, edge=edge_tag)
            elif len(faces) > 2:
                raise ValueError(f"Edge {edge_tag} is shared by more than 2 surfaces. The geometry seems to be a non-manifold.")

class FaceSegmenter:
    """ Class for segmenting a face-adjacency graph """
    def __init__(self, graph: nx.MultiDiGraph, has_fillet_radii: bool, has_flange: bool, bottom_angle_threshold: float = 30.0) -> None:
        """
        Initialize the FaceSegmenter

        :param graph: Input graph that gets segmented
        :param has_fillet_radii: If true fillet radii are added to potential segmentation labels.
        :param has_flange: If true flange faces are added to potential segmentation labels.
        :param bottom_angle_threshold: Angle difference threshold from starting bottom node to next bottom node.
        :return: None
        """
        self.graph = graph
        self.has_fillet_radii = has_fillet_radii
        self.has_flange = has_flange
        self.bottom_angle_threshold = bottom_angle_threshold
        self.positions = self._get_centroids()

    def segment_faces(self) -> dict:
        faces: dict = {}

        # 1. Bottom
        faces["bottom_faces"] = self._find_bottom_faces()

        # 2. Fillet Punch (optional)
        if self.has_fillet_radii:
            faces["fillet_punch_faces"] = self._adjacent(faces["bottom_faces"], exclude=faces["bottom_faces"])
            current_front = faces["fillet_punch_faces"]
        else:
            faces["fillet_punch_faces"] = list()
            current_front = faces["bottom_faces"]

        # 3. Slant (always)
        all_so_far = set(faces["bottom_faces"]).union(faces["fillet_punch_faces"])
        faces["slant_faces"] = self._adjacent(current_front, exclude=all_so_far)
        current_front = faces["slant_faces"]
        all_so_far = all_so_far.union(faces["slant_faces"])

        # 4. Fillet Die (optional)
        if self.has_fillet_radii:
            faces["fillet_die_faces"] = self._adjacent(current_front, exclude=all_so_far)
            current_front = faces["fillet_die_faces"]
            all_so_far = all_so_far.union(faces["fillet_die_faces"])
        else:
            faces["fillet_die_faces"] = list()

        # 5. Flange (optional)
        if self.has_flange:
            faces["flange_faces"] = self._adjacent(current_front, exclude=all_so_far)
        else:
            faces["flange_faces"] = list()

        return faces

    def _get_centroids(self) -> dict:
        return {
            node: np.array(data.get("center", (0.0, 0.0, 0.0)))
            for node, data in self.graph.nodes(data=True)
        }

    def _find_bottom_faces(self) -> list:
        start_node = self._find_start_node()
        bottom_nodes = [start_node]
        visited = {start_node}
        current = start_node

        while True:
            curr_pos = self.positions[current]
            neighbors = nx.all_neighbors(self.graph, current)

            if not neighbors:
                break

            # Select the neighbor most aligned with the X-axis
            best_neighbor = None
            smallest_angle = float("inf")

            for neighbor in neighbors:
                neighbor_pos = self.positions[neighbor]
                vec = neighbor_pos - curr_pos
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    continue
                cos_angle = np.dot(vec / norm, np.array([1.0, 0.0, 0.0]))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                if angle < smallest_angle:
                    smallest_angle = angle
                    best_neighbor = neighbor

            if best_neighbor is None or smallest_angle > self.bottom_angle_threshold:
                break

            bottom_nodes.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

        return list(bottom_nodes)

    def _find_start_node(self) -> int:
        # Get median y
        all_y = [v[1] for v in self.positions.values()]
        median_y = np.median(all_y)

        # Compute score for each point: low x + low z + y close to median
        def score(v):
            return v[0] + v[2] + abs(v[1] - median_y)

        # Get key with the lowest score
        best_key = min(self.positions.items(), key=lambda item: score(item[1]))[0]
        return best_key

    def _adjacent(self, nodes: set, exclude: set) -> list:
        result = set()
        for node in nodes:
            for neighbor in nx.all_neighbors(self.graph, node):
                if neighbor not in exclude:
                    result.add(neighbor)
        return list(result)

