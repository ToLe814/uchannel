from pathlib import Path
import sys
import math
from typing import List, Optional, Tuple, final
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gmsh
import yaml
import numpy as np
import json
import networkx as nx
import itertools
from collections import defaultdict

from uchannel.error_handling import FeasibilityError, FilletError, RemoveHelperEdgesError, FragmentError
from uchannel.segmentation import FaceAdjacencyGraph, FaceSegmenter
from uchannel.functions import set_axes_equal
from uchannel.mathematics import tangent_from_corner, unit, cross, dot, add, smul, clamp, distance, sub, split_by_centerline


class UChannelBase:
    """
    Base class for all UChannel classes. This class provides the basic structure.
    """
    def __init__(self, geom_path: Path, label_path: Path, **kwargs):
        self.kwargs = kwargs
        self.geom_path = geom_path
        self.label_path = label_path
        self.model = None
        self.fillet_method = 'explicit' # 'explicit' or 'gmsh' or 'none'
        self.bounding_box = None

        # Fillet radius placeholder
        self.r: float = 0.0

        # Addendum and flange placeholder
        self.ad: bool = None
        self.fl: bool = None

        self.above, self.below = None, None

        # Segmentation
        self.face_adjacency = FaceAdjacencyGraph()
        self.face_segmenter = None
        self.segmentation_labels: dict = {}

        # Define named tuples for the vertices
        self.Vertex = namedtuple('Vertex', ['x', 'y', 'z'])
        self.Bounds = namedtuple("bounds", ["min", "max"])
        self.surface_labels = {}

        # Define graph. This is used for explicit filleting.
        self.graph = nx.Graph()

    @final
    def _check_fillet_method(self) -> None:
        """ Check if the defined fillet method is valid."""
        if self.fillet_method not in ['explicit', 'gmsh', 'none']:
            raise ValueError(f"Invalid fillet method: {self.fillet_method}. UChannel only supports 'explicit', 'gmsh', or 'none'.")

    @final
    def _create_surfaces(self) -> None:
        """ Wrapper for the surface creation method. """
        if self.fillet_method == 'explicit':
            self._create_surfaces_explicit()
        else:
            self._create_surfaces_procedural()

    @staticmethod
    def _create_surfaces_explicit() -> None:
        """ Placeholder for the explicit surface creation which is to be implemented in the subclasses. """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def _create_surfaces_procedural() -> None:
        """
        Placeholder for the procedural surface creation (when self.fillet_method in ['gmsh', 'none'])
        which is to be implemented in the subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @final
    def _clean_geometry(self, dim_tags: List[Tuple[int, int]] = None) -> None:
        """ As a final step in geometry creation, remove all points and curves. This removes only the unused ones. """
        all_point_tags = self.model.occ.getEntities(0)
        self.model.occ.remove(all_point_tags, recursive=True)

        all_curve_tags = self.model.occ.getEntities(1)
        self.model.occ.remove(all_curve_tags)

        if dim_tags:
            self.model.occ.remove(dim_tags)

        self.model.occ.synchronize()

    @staticmethod
    def _create_fillet(edges: List[int], radii: List[float]) -> None:
        """
        Create a fillet on the given edges with the given radius. The radii vector can either contain a single radius,
        as many radii as edges, or twice as many as edges (in which case different radii are provided for the start and
        end points of the curves)
        :param edges: curveTags of the edges to fillet
        :param radii: radii of the fillets
        :return: model with fillets
        """
        # Generate a volume from the surfaces
        sf_tags = [tag for dim, tag in gmsh.model.occ.getEntities(2)]
        sl_tag = gmsh.model.occ.addSurfaceLoop(sf_tags, 1)
        v_tag = gmsh.model.occ.addVolume([sl_tag], 1)

        # Create the fillet and remove the volume
        if len(radii) == 1 and radii[0] in [0, 0.0]:  # If there is a single radius of 0 for every edge, do not fillet.
            pass
        elif sum(radii) == 0:  # If there are no radii, do not fillet.
            pass
        else:
            try:
                gmsh.model.occ.fillet([v_tag], edges, radii, removeVolume=True)
            except Exception as e:
                raise FilletError(f'Error while creating fillet: {e}')

    @final
    @staticmethod
    def _validate_parameter(parameter_name: str | Tuple[str, str], value: float | Tuple[float, float],
                            min_value: float, max_value: float) -> None:
        """
        Validate the given parameter value. If the value is not within the given range, raise a FeasibilityError.
        :param parameter_name: name of the parameter
        :param value: value of the parameter
        :param min_value: minimum value of the parameter
        :param max_value: maximum value of the parameter
        :return: None
        """
        if isinstance(parameter_name, tuple) and isinstance(value, tuple):
            symmetric_pairs = {
                ("r_l[0]", "r_r[0]"),
                ("r_l[1]", "r_r[1]"),
                ("r_r[0]", "r_l[0]"),
                ("r_r[1]", "r_l[1]"),
            }

            if tuple(parameter_name) in symmetric_pairs:
                zero_vals = [0, 0.0]
                val0_is_zero = value[0] in zero_vals
                val1_is_zero = value[1] in zero_vals

                if val0_is_zero != val1_is_zero:  # XOR: only one is zero
                    raise FeasibilityError(f"{parameter_name} must be 0 for both")
                else:
                    return
        if parameter_name in ["r_r", "r_l"]:
            if isinstance(value, tuple):
                if value[0] in [0.0, 0] and value[1] not in [0.0, 0]:
                    raise FeasibilityError(f"{parameter_name} must be 0 for both")
                if value[0] not in [0.0, 0] and value[1] in [0.0, 0]:
                    raise FeasibilityError(f"{parameter_name} must be 0 for both")
                else:
                    return

        if not (min_value <= value <= max_value):
            if "alpha" in parameter_name or "beta" in parameter_name:  # angles
                raise FeasibilityError(f'{parameter_name} must be between {math.degrees(min_value)} and'
                                       f' {math.degrees(max_value)}.')
            elif parameter_name in ["r_r[0]", "r_r[1]", "r_l[0]", "r_l[1]", "r"] and value in [0, 0.0]:  # radii
                # Omit bounds check for sharp edges
                pass
            else:
                raise FeasibilityError(f'{parameter_name} must be between {min_value} and {max_value}.')

    def check_feasibility(self) -> None:
        """ Check if the parameters are feasible. """
        raise NotImplementedError("Subclasses must implement this method.")

    def _export_geometry(self) -> None:
        """ Export the geometry to the specified path."""
        gmsh.write(str(self.geom_path)) # pathlib path has to be converted to str as gmsh doesn't accept them otherwise.

    def _export_segmentation_labels(self) -> None:
        """ Export the segmentation labels to a JSON file using the gmsh face idx. """
        with open(self.label_path, 'w') as f:
            json.dump(self.segmentation_labels, f, indent=4)

    @final
    @staticmethod
    def export_parameter(params: dict, output_path: Path) -> None:
        """
        Save parameters as YAML, converting NumPy types into plain Python types.
        """

        def clean_value(value):
            # Handle NumPy scalars
            if isinstance(value, np.generic):
                return value.item()
            # Handle NumPy arrays
            if isinstance(value, np.ndarray):
                return value.tolist()
            # Handle booleans
            if isinstance(value, (bool, np.bool_)):
                return "True" if value else "False"
            # Handle tuples
            if isinstance(value, tuple):
                return [clean_value(v) for v in value]
            # Handle lists
            if isinstance(value, list):
                return [clean_value(v) for v in value]
            # Handle dicts (nested)
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            return value

        cleaned_params = {k: clean_value(v) for k, v in params.items()}

        with open(output_path, "w") as yaml_file:
            yaml.safe_dump(cleaned_params, yaml_file, indent=4, sort_keys=False)

    @staticmethod
    def export_fillet_edges(edges: List, output_path: Path) -> None:
        """
        Export the fillet vertices list to a txt file.
        :param edges: List of edges
        :param output_path: File path to save the txt file
        :return: None
        """
        if output_path.suffix != '.txt':
            raise ValueError(f'Output path must be a txt file. Got: {output_path}')

        with open(output_path, 'w') as file:
            file.write(str(edges))

    @staticmethod
    def _visualize_geometry() -> None:
        """ Wrapper for geometry visualization in gmsh gui. """
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()

    def create_geometry(self):
        """ Placeholder for geometry creation. To be implemented in the subclasses. """
        raise NotImplementedError("Subclasses must implement this method.")

    @final
    @staticmethod
    def __add_corner(
            edge_a_tag: int,
            edge_b_tag: int,
            corner_tag: int,
            radius: float,
            adjust_radius: bool = False,
            a_1_point: Optional[Tuple[float, float, float]] = None
    ) -> Optional[Tuple[int, int, int]]:
        """
        Create a corner (circular arc) between two edges at a given corner point.

        :param edge_a_tag: Tag of the first edge
        :param edge_b_tag: Tag of the second edge
        :param corner_tag: Tag of the corner point
        :param radius: Radius of the fillet
        :param adjust_radius: Whether to adjust the radius to be within the bounds of the edges
        :param a_1_point: Optional explicit point on edge A where the arc should start
        :return: Tuple of the arc tag, the tag of the first point of the arc, and the tag of the second point of the arc
        """
        corner_point: Tuple[float, float, float] = tuple(gmsh.model.getValue(0, corner_tag, []))
        t1 = tangent_from_corner(edge_a_tag, corner_point)
        t2 = tangent_from_corner(edge_b_tag, corner_point)

        n = unit(cross(t1, t2))
        cos_th = clamp(dot(t1, t2), -1.0, 1.0)
        theta = math.acos(cos_th)
        if theta < 1e-2 or abs(theta - math.pi) < 1e-2:
            return None

        k1 = unit(cross(n, t1))
        k2 = unit(cross(t2, n))
        if dot(k1, k2) < 0.0:
            k2 = smul(-1.0, k2)

        # Adjust radius if necessary
        if adjust_radius:
            min_radius = 1e-3
            user_defined_radius = radius

            def edge_length_from_corner(tag, corner_point):
                pt_tags = gmsh.model.getAdjacencies(1, tag)[1]
                pt_coords = [tuple(gmsh.model.getValue(0, pt, [])) for pt in pt_tags]
                dists = [distance(corner_point, pt) for pt in pt_coords]
                return max(dists)

            len_a = edge_length_from_corner(edge_a_tag, corner_point)
            len_b = edge_length_from_corner(edge_b_tag, corner_point)
            max_l = min(len_a, len_b) * 0.95
            max_radius = max_l * math.tan(0.5 * theta)

            if radius > max_radius:
                radius = max_radius

            if radius < min_radius:
                radius = user_defined_radius

        # Distance along tangents
        l = radius / math.tan(0.5 * theta)
        cdist = radius / math.sin(0.5 * theta)
        sdir = unit(add(k1, k2))

        if a_1_point is None:
            # === OLD LOGIC ===
            a_1 = add(corner_point, smul(l, t1))
            a_2 = add(corner_point, smul(l, t2))
            c = add(corner_point, smul(cdist, sdir))

            if distance(a_1, a_2) < 1e-6:
                return None

            # Add helper points and create the circular arc
            point_tag_a1 = gmsh.model.occ.addPoint(*a_1)
            point_tag_c = gmsh.model.occ.addPoint(*c)
            point_tag_a2 = gmsh.model.occ.addPoint(*a_2)
            line_tag_arc = gmsh.model.occ.addCircleArc(point_tag_a1, point_tag_c, point_tag_a2)

            gmsh.model.occ.remove([(0, point_tag_c)], recursive=True)
            return line_tag_arc, point_tag_a1, point_tag_a2
        else:
            a_1 = gmsh.model.getValue(0, a_1_point, [])

            # === NEW LOGIC (user supplies a_1) ===
            # 1) Get endpoints (0D "downward" adjacencies) of the edge
            up, down = gmsh.model.getAdjacencies(1, edge_b_tag)  # up: dim+1, down: dim-1 (points)

            endpoints = list(down)
            # 2) Pick the endpoint that's NOT the corner point
            other_pt_tag = next(t for t in endpoints if t != corner_tag)

            # 3) Get coordinates
            corner = gmsh.model.getValue(0, corner_tag, [])
            other = gmsh.model.getValue(0, other_pt_tag, [])

            # 4) Direction from corner -> other
            a_2_dir = unit(sub(other, corner))

            # 5) Place a point l units away from corner in that direction
            a_2 = add(corner, smul(l, a_2_dir))

            # 6) Compute bisector direction between (a1 - corner) and (a2 - corner)
            v1 = unit(sub(a_1, corner))
            v2 = unit(sub(a_2, corner))
            sdir = unit(add(v1, v2))

            if distance(a_1, a_2) < 1e-6:
                return None

            # 7) Compute middle point c at distance cdist from corner
            c = add(corner, smul(cdist, sdir))
            r1 = distance(a_1, c)
            r2 = distance(a_2, c)

            # vectors from center to endpoints
            u1 = unit(sub(a_1, c))
            u2 = unit(sub(a_2, c))

            # arc radius (use the average to be robust)
            r = 0.5 * (distance(a_1, c) + distance(a_2, c))

            # mid-arc direction (shorter arc)
            u_mid = unit(add(u1, u2))  # fails only if angle ~180°
            p_mid = add(c, smul(r, u_mid))  # <- point on the arc

            # Add helper points and create the circular arc
            point_tag_p_mid = gmsh.model.occ.addPoint(*p_mid)
            point_tag_a2 = gmsh.model.occ.addPoint(*a_2)
            line_tag_arc = gmsh.model.occ.addSpline([a_1_point, point_tag_p_mid, point_tag_a2])
            gmsh.model.occ.remove([(0, point_tag_p_mid)], recursive=True)

            return line_tag_arc, a_1_point, point_tag_a2

    @final
    def _fragment_at_graph_nodes(self) -> None:
        """
        Fragment each selected edge at all points in the model.
        - If 2 fragments:
            * If splitting node has `is_flange="is_flange"`, delete the one closer to Y center.
            * Otherwise delete the shorter one.
        - If 3 fragments: delete left and right, keep the middle.
        """

        def _edge_coords(etag):
            """Get coordinates of edge endpoints."""
            _, pts = gmsh.model.getAdjacencies(1, etag)
            return [tuple(gmsh.model.getValue(0, p, [])) for p in pts]

        def _edge_midpoint(etag):
            coords = _edge_coords(etag)
            if not coords:
                return (0.0, 0.0, 0.0)
            return tuple(sum(p[i] for p in coords) / len(coords) for i in range(3))

        def _edge_length(etag):
            coords = _edge_coords(etag)
            if len(coords) == 2:
                (x1, y1, z1), (x2, y2, z2) = coords
                return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
            return 0.0


        all_edges = gmsh.model.getEntities(1)
        for tag in all_edges:
            all_points = gmsh.model.getEntities(0)
            try:
                _, mapping = self.model.occ.fragment([tag], all_points)
                # update graph
                self._rebuild_graph_from_occ()
                self.model.occ.synchronize()
            except Exception as e:
                raise FragmentError(f"Error while fragmenting edge {tag}: {e}")

            if not mapping:
                continue

            # gather new curve tags from mapping
            curves = [obj for obj in mapping[0] if obj[0] == 1]
            new_edges = [c[1] for c in curves]

            # --- Case: 2 fragments
            if len(new_edges) == 2:
                # check if the new edges are in line with the yz plane
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(1, new_edges[0])
                in_plane_a = abs(xmin - xmax) < 1e-6
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(1, new_edges[1])
                in_plane_b = abs(xmin - xmax) < 1e-6

                if in_plane_a and in_plane_b:
                    in_plane = True
                else:
                    in_plane = False

                if in_plane:
                    # delete the fragment that lies closer to the "split reference"
                    mids = [_edge_midpoint(e) for e in new_edges]

                    # wrap mids as synthetic node-like dicts so we can pass through _split_by_reference
                    fake_nodes = []
                    for i, (mx, my, mz) in enumerate(mids):
                        nid = f"_mid_{tag}_{i}"
                        self.graph.add_node(nid, coords=(mx, my, mz))
                        fake_nodes.append(nid)

                    below, above = self._split_by_reference(fake_nodes)

                    if below and above:
                        # Normal case: both groups non-empty → kill the minority
                        if len(below) < len(above):
                            kill_index = fake_nodes.index(below[0])
                        else:
                            kill_index = fake_nodes.index(above[0])
                    else:
                        # Fallback: all points fell on one side → choose the one closest to the reference line
                        mids = [self.graph.nodes[n]["coords"] for n in fake_nodes]
                        if self.__class__.__name__ == "UChannel1":
                            # distance to centerline (p1..p2)
                            import numpy as np
                            p1 = np.array((0, self.y_l / 2, 0))
                            p2 = np.array(((self.x_1 + self.x_2) / 2, self.y_r / 2, 0))
                            line_vec = p2 - p1

                            def dist(pt):
                                pt_vec = np.array(pt) - p1
                                return np.linalg.norm(np.cross(line_vec, pt_vec)) / np.linalg.norm(line_vec)
                        else:
                            # distance to horizontal y_center
                            if self.bounding_box[1] < self.bounding_box[4]:
                                y_center = (self.bounding_box[1] + self.bounding_box[4]) / 2
                            else:
                                y_center = (self.bounding_box[4] + self.bounding_box[1]) / 2

                            def dist(pt):
                                return abs(pt[1] - y_center)

                        dists = [dist(m) for m in mids]
                        kill_index = int(min(range(len(dists)), key=lambda i: dists[i]))

                    kill = new_edges[kill_index]

                    # clean up fake nodes
                    self.graph.remove_nodes_from(fake_nodes)
                else:
                    # delete the shorter one
                    l1, l2 = _edge_length(new_edges[0]), _edge_length(new_edges[1])
                    kill = new_edges[0] if l1 < l2 else new_edges[1]

                self.model.occ.remove([(1, kill)], recursive=False)
                self.graph.remove_edges_from(
                    [(uu, vv) for uu, vv, d in self.graph.edges(data=True) if d.get("tag") == kill]
                )

            # --- Case: 3 fragments
            elif len(new_edges) == 3:
                # check if the new edges are in line with the yz plane
                in_plane_yz = False
                for e in new_edges:
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(1, e)
                    in_plane_yz = abs(xmin - xmax) < 1e-6

                mids = []
                for e in new_edges:
                    mx, my, mz = _edge_midpoint(e)
                    if in_plane_yz:
                        mids.append((mz, e))
                    else:
                        mids.append((mx, e))

                mids.sort(key=lambda x: x[0])  # left..middle..right
                left, middle, right = [e for _, e in mids]

                self.model.occ.remove([(1, left), (1, right)], recursive=False)
                self.graph.remove_edges_from(
                    [(uu, vv) for uu, vv, d in self.graph.edges(data=True) if d.get("tag") in (left, right)]
                )

    def __edge_vector(self, node, edge) -> np.ndarray:
        """
        Calculate the normalized direction vector of an edge at a node.
        :param node: Node tag
        :param edge: Edge tuple (u, v, data)
        :return: Normalized direction vector.
        """
        u, v, data = edge
        other = v if u == node else u
        p0 = np.array(self.graph.nodes[node]["coords"])
        p1 = np.array(self.graph.nodes[other]["coords"])
        vec = p1 - p0
        return vec / np.linalg.norm(vec)

    def _graph_corner(self, flange_radii_tags: List[int], tolerance: Optional[float] = 1e-8,
                      radii: Optional[List[float]] = None) -> None:
        """
        Create corners (circular arcs) at nodes with graph degree 3 and 4.

        :param flange_radii_tags: List of flange radii tags
        :param tolerance: Tolerance for edge alignment
        :param radii: List of radii for the corners
        :return: None
        """
        degree_3_counter: int = 0

        for node in list(self.graph.nodes):
            deg = self.graph.degree(node)
            is_flange = (
                "is_flange" if self.graph.nodes[node]["tag"] in flange_radii_tags else None
            )

            if deg == 3:
                degree_3_counter = self.__handle_degree_3(
                    node=node, tolerance=tolerance, radii=radii, degree_3_counter=degree_3_counter,is_flange=is_flange
                )

            elif deg == 4:
                self.__handle_degree_4(node=node, is_flange=is_flange)

    def __handle_degree_3(self, node: str, tolerance: float, radii: Optional[List[float]], degree_3_counter: int,
                          is_flange: Optional[str]) -> int:
        """
        Add fillet to nodes with degree 3
        :param node: Node tag
        :param tolerance: Tolerance for edge alignment
        :param radii: Optional list of radii for the corners
        :param degree_3_counter: Counter for degree 3 nodes
        :param is_flange: Optional flange tag
        :return: Counter for degree 3 nodes
        """
        edges = list(self.graph.edges(node, data=True))

        yz_edges = self.__edges_in_yz_plane(node, tolerance=tolerance)
        if len(yz_edges) == 2:
            e1, e2 = yz_edges
            suffix = "yz"
        else:
            e1, e2 = self.__get_most_collinear_pair(node, edges)
            suffix = "collinear"
            # if not self.fl:
            #     return degree_3_counter

        pt_tag = self.graph.nodes[node]["tag"]
        radius = radii[degree_3_counter] if radii else self.r
        degree_3_counter += 1

        result = self.__add_corner(
            edge_a_tag=e1[2]["tag"], edge_b_tag=e2[2]["tag"], corner_tag=pt_tag, radius=radius, adjust_radius=True
        )
        if result is None:
            return degree_3_counter

        self.model.occ.synchronize()
        arc_tag, p_start, p_end = result
        coords_start = tuple(gmsh.model.getValue(0, p_start, []))
        coords_end = tuple(gmsh.model.getValue(0, p_end, []))

        if suffix == "collinear":
            side_start, side_end = (
                ("left", "right") if coords_start[0] <= coords_end[0] else ("right", "left")
            )
        elif suffix == "yz":
            side_start, side_end = (
                ("bottom", "top") if coords_start[2] <= coords_end[2] else ("top", "bottom")
            )
        else:
            raise ValueError(f"Unknown suffix: {suffix}")

        n_start = self.__make_node_name(node, "fillet", suffix, side_start, is_flange=is_flange)
        n_end = self.__make_node_name(node, "fillet", suffix, side_end, is_flange=is_flange)

        self.graph.add_node(
            n_start, tag=p_start, kind="fillet", subtype=suffix, side=side_start, coords=coords_start, source_degree=3,
            base=node, is_flange=is_flange
        )
        self.graph.add_node(
            n_end, tag=p_end, kind="fillet", subtype=suffix, side=side_end, coords=coords_end, source_degree=3,
            base=node,is_flange=is_flange
        )
        self.graph.add_edge(n_start, n_end, name=f"arc_{node}_{suffix}", tag=arc_tag, kind="arc")

        return degree_3_counter

    def __handle_degree_4(self, node: str, is_flange: Optional[str]) -> None:
        """
        Add fillet to nodes with degree 4

        :param node: Node tag
        :param is_flange: Flange tag
        :return: None
        """
        edges = list(self.graph.edges(node, data=True))

        # Fillet yz-plane pair
        yz_edges = self.__edges_in_yz_plane(node)
        if len(yz_edges) == 2:
            e1, e2 = yz_edges
            pt_tag = self.graph.nodes[node]["tag"]
            result = self.__add_corner(edge_a_tag=e1[2]["tag"], edge_b_tag=e2[2]["tag"], corner_tag=pt_tag,
                                       radius=self.r, adjust_radius=True)
            if result is not None:
                self.model.occ.synchronize()
                arc_tag, p_start, p_end = result
                coords_start = tuple(gmsh.model.getValue(0, p_start, []))
                coords_end = tuple(gmsh.model.getValue(0, p_end, []))
                side_start, side_end = (
                    ("bottom", "top") if coords_start[2] <= coords_end[2] else ("top", "bottom")
                )

                n_start = self.__make_node_name(node, "fillet", "yz", side_start, is_flange=is_flange)
                n_end = self.__make_node_name(node, "fillet", "yz", side_end, is_flange=is_flange)

                self.graph.add_node(
                    n_start, tag=p_start, kind="fillet", subtype="yz", side=side_start, coords=coords_start,
                    source_degree=4, base=node, is_flange=is_flange
                )
                self.graph.add_node(
                    n_end, tag=p_end, kind="fillet", subtype="yz", side=side_end, coords=coords_end, source_degree=4,
                    base=node, is_flange=is_flange
                )

                self.graph.add_edge(
                    n_start, n_end, name=f"arc_{node}_yz", tag=arc_tag, kind="arc", subtype="yz", is_helper=True
                )

            # Fillet the remaining edges
            remaining = [e for e in edges if e not in yz_edges]

            if self.ad and self.graph.nodes[node]["tag"] in [3, 4]:
                self.graph.nodes[n_start]["kind"] = "junction"
                self.graph.nodes[n_start]["subtype"] = "junction"
                self.graph.nodes[n_start]["side"] = "special"
                self.graph.nodes[n_start]["vert_side"] = side_start
                self.graph.nodes[n_end]["kind"] = "junction"
                self.graph.nodes[n_end]["subtype"] = "junction"
                self.graph.nodes[n_end]["side"] = "special"
                self.graph.nodes[n_end]["vert_side"] = side_end
                remaining = []

            if len(remaining) == 2:
                e3, e4 = remaining
                result = self.__add_corner(
                    edge_a_tag=e3[2]["tag"], edge_b_tag=e4[2]["tag"], corner_tag=pt_tag, radius=self.r,
                    adjust_radius=False
                )
                if result is not None:
                    self.model.occ.synchronize()
                    arc_tag, p_start, p_end = result
                    coords_start = tuple(gmsh.model.getValue(0, p_start, []))
                    coords_end = tuple(gmsh.model.getValue(0, p_end, []))
                    side_start, side_end = (
                        ("left", "right") if coords_start[0] <= coords_end[0] else ("right", "left")
                    )

                    n_start = self.__make_node_name(node, "fillet", "rem", side_start, is_flange=is_flange)
                    n_end = self.__make_node_name(node, "fillet", "rem", side_end, is_flange=is_flange)

                    self.graph.add_node(
                        n_start, tag=p_start, kind="fillet", subtype="rem", side=side_start, coords=coords_start,
                        source_degree=4, base=node, is_flange=is_flange
                    )
                    self.graph.add_node(
                        n_end, tag=p_end, kind="fillet", subtype="rem", side=side_end, coords=coords_end,
                        source_degree=4, base=node, is_flange=is_flange
                    )
                    self.graph.add_edge(
                        n_start, n_end, name=f"arc_{node}_rem", tag=arc_tag, kind="arc", subtype="rem", is_helper=True
                    )

                else:
                    self.graph.nodes[n_start]["kind"] = "junction"
                    self.graph.nodes[n_start]["subtype"] = "junction"
                    self.graph.nodes[n_start]["side"] = "special"
                    vert = "top" if "top" in n_start else "bottom"
                    self.graph.nodes[n_start]["vert_side"] = vert
                    self.graph.nodes[n_end]["kind"] = "junction"
                    self.graph.nodes[n_end]["subtype"] = "junction"
                    self.graph.nodes[n_end]["side"] = "special"
                    vert = "bottom" if "bottom" in n_end else "top"
                    self.graph.nodes[n_end]["vert_side"] = vert

                    self.graph.edges[e1[0], e1[1]]["is_helper"] = False
                    self.graph.edges[e2[0], e2[1]]["is_helper"] = False

                    for u, v, d in self.graph.edges(data=True):
                        if d.get("kind") == "arc":
                            if d.get("name").split("_")[-2] in e1[0] or d.get("name").split("_")[-2] in e2[0]:
                                self.graph.edges[u, v]["is_helper"] = False

    @final
    def _remove_helper_edges(self) -> None:
        """
        Remove all edges from the graph (and OCC) that are marked as helper edges.
        An edge is considered helper if it has attribute is_helper=True.
        """
        if not hasattr(self, "graph") or self.graph is None:
            raise RuntimeError("No graph available. Did you call _create_vertices() and _create_edges()?")

        edges_to_remove = [
            (u, v, d)
            for u, v, d in self.graph.edges(data=True)
            if d.get("is_helper", False) is True
        ]

        for u, v, d in edges_to_remove:
            tag = d.get("tag")
            if tag is not None:
                try:
                    # Remove from gmsh
                    self.model.occ.remove([(1, tag)], recursive=True)
                except Exception as e:
                    raise RemoveHelperEdgesError(f"Failed to remove helper edge {tag}: {e}")

            # Remove from graph
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
            if self.graph.has_node(u):
                self.graph.remove_node(u)
            if self.graph.has_node(v):
                self.graph.remove_node(v)

        self.model.occ.synchronize()

    @final
    def _get_bounding_box(self) -> None:
        """ Calculate the bounding box of the geometry and store it in self.bounding_box """
        self.bounding_box = self.model.getBoundingBox(-1, -1)

    @final
    def _connect_fillet_nodes_on_yz_planes(
            self, tolerance: float = 3.0, angle_tol_deg: float = 3.0
    ) -> None:
        """
        Connect fillet nodes into vertical chains on YZ-planes.
        If exactly 3 edges are created and they are nearly collinear:
          - delete the middle edge,
          - create two new edges from the middle node, connecting to
            the 'bottom' yz-plane fillet nodes that share the base name
            of the outer edge start nodes.
        """

        def _yz_vec(u, v):
            nu = self.graph.nodes[u]["coords"]
            nv = self.graph.nodes[v]["coords"]
            return (nv[1] - nu[1], nv[2] - nu[2])

        def _angle_deviation_deg(v1, v2):
            (y1, z1), (y2, z2) = v1, v2
            n1 = math.hypot(y1, z1)
            n2 = math.hypot(y2, z2)
            if n1 == 0 or n2 == 0:
                return 0.0
            cosang = (y1 * y2 + z1 * z2) / (n1 * n2)
            cosang = max(-1.0, min(1.0, cosang))
            ang = math.degrees(math.acos(cosang))
            return min(ang, 180.0 - ang)

        def _find_bottom_fillet(base_node, rename: bool = True):
            """Find node with same base name, kind=fillet, subtype=yz, side=bottom."""
            base = str(base_node).split("_")[0]  # adjust if naming differs
            for n, data in self.graph.nodes(data=True):
                if (
                        str(n).startswith(base)
                        and data.get("kind") == "fillet"
                        and data.get("side") == "bottom"
                ):
                    self.model.occ.synchronize()
                    line_tag = self.model.occ.addLine(
                        self.graph.nodes[base_node]["tag"], self.graph.nodes[n]["tag"]
                    )
                    if rename:
                        self.graph.nodes[n]["rename"] = True

                    return line_tag
            return None

        def _find_top_fillet(base_node, rename: bool = True):
            """Find node with same base name, kind=fillet, subtype=yz, side=top."""
            base = str(base_node).split("_")[0]  # adjust if naming differs
            for n, data in self.graph.nodes(data=True):
                if (
                        str(n).startswith(base)
                        and data.get("kind") == "fillet"
                        and data.get("side") == "top"
                ):
                    self.model.occ.synchronize()
                    line_tag = self.model.occ.addLine(
                        self.graph.nodes[base_node]["tag"], self.graph.nodes[n]["tag"]
                    )
                    if rename:
                        self.graph.nodes[n]["rename"] = True

                    return line_tag
            return None

        # --- step 1: collect fillet nodes
        fillet_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get("kind") == "fillet"
               and data.get("subtype") in ("rem", "collinear")
        ]
        if not fillet_nodes:
            return

        # --- step 2: group by x and side
        groups = {}
        for node in fillet_nodes:
            data = self.graph.nodes[node]
            x = data["coords"][0]
            side = data.get("side", "unknown")
            key = (round(x / tolerance) * tolerance, side)
            groups.setdefault(key, []).append(node)

        # --- step 3: connect within each group
        edges = []
        for (xkey, side), nodes in groups.items():
            if len(nodes) < 2:
                continue

            # sort nodes by y,z to form chain
            nodes_sorted = sorted(
                nodes,
                key=lambda n: (
                    self.graph.nodes[n]["coords"][1],
                    self.graph.nodes[n]["coords"][2],
                ),
            )

            created_edges = []
            for i, u in enumerate(nodes_sorted[:-1]):
                uy = self.graph.nodes[u]["coords"][1]

                # find candidate v on the same side with minimum y deviation
                candidates = [
                    v for v in nodes_sorted[i + 1:]
                    if self.graph.nodes[v].get("side") == self.graph.nodes[u].get("side")
                ]
                if not candidates:
                    continue

                v = min(
                    candidates,
                    key=lambda v: abs(self.graph.nodes[v]["coords"][1] - uy)
                )

                tag = self.model.occ.addLine(
                    self.graph.nodes[u]["tag"], self.graph.nodes[v]["tag"]
                )
                self.graph.add_edge(
                    u, v, name=f"support_{u}_{v}", tag=tag, kind="support"
                )
                created_edges.append({"u": u, "v": v, "tag": tag, "vec": _yz_vec(u, v)})

            if created_edges:
                edges.append(created_edges)

        left_right_to_be_refeatured = []
        to_be_refeatured = []
        for created_edges in edges:

            if len(created_edges) == 1:
                left = created_edges[0]["u"]
                right = created_edges[0]["v"]

                zmin = self.bounding_box[2]
                zmax = self.bounding_box[5]
                z = self.graph.nodes[left]["coords"][2]
                if z > (zmax + zmin)*3/4:
                    self.model.occ.remove([(1, created_edges[0]["tag"])], recursive=False)
                    if self.graph.has_edge(created_edges[0]["u"], created_edges[0]["v"]):
                        self.graph.remove_edge(created_edges[0]["u"], created_edges[0]["v"])

                    continue

                # --- LEFT SIDE ---
                left_line_tag = _find_top_fillet(left, rename=True)
                self.model.occ.synchronize()
                _, pt_tags = self.model.getAdjacencies(1, left_line_tag)
                top_fillet_tag = max(pt_tags, key=lambda pt: self.model.getValue(0, pt, [])[2])
                top_node = [node for node, data in self.graph.nodes(data=True) if data.get("tag") == top_fillet_tag][0]
                to_be_refeatured.append(top_node)
                corner = min(pt_tags, key=lambda pt: self.model.getValue(0, pt, [])[2])
                corner = self.__add_corner(
                    left_line_tag,
                    created_edges[0]["tag"],
                    corner,
                    self.r,
                    a_1_point=top_fillet_tag,
                )
                _, _, p_end = corner
                self.model.occ.synchronize()
                coord = self.model.getValue(0, p_end, [])

                side = "left"

                # check if there already exists a junction with the same base
                for node, data in self.graph.nodes(data=True):
                    if data.get("kind") == "junction" and data.get("base") == self.graph.nodes[left][
                        "base"] and data.get("vert_side") == "bottom":
                        if data.get("tag") == self.graph.nodes[left]["tag"]:
                            continue
                        coordinates = data.get("coords")
                        # if new coordinates are >x than existing coordinates, reset the side
                        if coordinates[0] < coord[0]:
                            self.graph.nodes[node]["side"] = "right"
                            side = "left"
                        else:
                            self.graph.nodes[node]["side"] = "left"
                            side = "right"

                self.graph.add_node(
                    f"test_{p_end}",
                    tag=p_end,
                    kind="junction",
                    subtype="junction",
                    side=side,
                    vert_side="bottom",
                    source_degree="junction",
                    base=self.graph.nodes[left]["base"],
                    is_flange=None,
                    coords=coord,
                )
                left_right_to_be_refeatured.append(f"test_{p_end}")

                # --- RIGHT SIDE ---
                right_line_tag = _find_top_fillet(right, rename=True)
                self.model.occ.synchronize()
                _, pt_tags = self.model.getAdjacencies(1, right_line_tag)
                top_fillet_tag = max(pt_tags, key=lambda pt: self.model.getValue(0, pt, [])[2])
                top_node = [node for node, data in self.graph.nodes(data=True) if data.get("tag") == top_fillet_tag][0]
                to_be_refeatured.append(top_node)
                corner = min(pt_tags, key=lambda pt: self.model.getValue(0, pt, [])[2])
                corner = self.__add_corner(
                    right_line_tag,
                    created_edges[0]["tag"],
                    corner,
                    self.r,
                    a_1_point=top_fillet_tag,
                )
                _, _, p_end = corner
                self.model.occ.synchronize()
                coord = self.model.getValue(0, p_end, [])

                side = "right"

                # check if there already exists a junction with the same base
                for node, data in self.graph.nodes(data=True):
                    if data.get("kind") == "junction" and data.get("base") == self.graph.nodes[right][
                        "base"] and data.get("vert_side") == "bottom":
                        if data.get("tag") == self.graph.nodes[right]["tag"]:
                            continue
                        coordinates = data.get("coords")
                        if coordinates[0] < coord[0]:
                            self.graph.nodes[node]["side"] = "left"
                            side = "right"
                        else:
                            self.graph.nodes[node]["side"] = "right"
                            side = "left"

                self.graph.add_node(
                    f"test_{p_end}",
                    tag=p_end,
                    kind="junction",
                    subtype="junction",
                    side=side,
                    vert_side="bottom",
                    source_degree="junction",
                    base=self.graph.nodes[right]["base"],
                    is_flange=None,
                    coords=coord,
                )
                left_right_to_be_refeatured.append(f"test_{p_end}")

            if len(created_edges) == 3:
                e1, e2, e3 = created_edges
                dev12 = _angle_deviation_deg(e1["vec"], e2["vec"])
                dev23 = _angle_deviation_deg(e2["vec"], e3["vec"])

                if dev12 < angle_tol_deg and dev23 < angle_tol_deg:
                    # delete middle edge
                    u_mid, v_mid, tag_mid = e2["u"], e2["v"], e2["tag"]
                    self.model.occ.remove([(1, tag_mid)], recursive=False)
                    if self.graph.has_edge(u_mid, v_mid):
                        self.graph.remove_edge(u_mid, v_mid)

                    n_center = v_mid  # middle node is anchor

                    # find targets for left and right
                    left_line_tag = _find_bottom_fillet(e1["v"], rename=True)
                    self.model.occ.synchronize()
                    _, pt_tags = self.model.getAdjacencies(1, left_line_tag)
                    bottom_fillet_tag = min(pt_tags, key=lambda pt: self.model.getValue(0, pt, [])[2])
                    corner = self.__add_corner(left_line_tag, e1["tag"], self.graph.nodes[e1["v"]]["tag"], self.r, a_1_point=bottom_fillet_tag)
                    _, _, p_end = corner
                    self.model.occ.synchronize()
                    coord = self.model.getValue(0, p_end, [])

                    side = "left"

                    # check if there already exists a junction with the same base
                    for node, data in self.graph.nodes(data=True):
                        if data.get("tag") == self.graph.nodes[e1["v"]]["tag"]:
                            continue
                        if data.get("kind") == "junction" and data.get("base") == self.graph.nodes[e1["v"]]["base"] and data.get("vert_side") == "top":
                            coordinates = data.get("coords")
                            # if new coordinates are >x than existing coordinates, reset the side to right
                            if coordinates[0] < self.model.getValue(0, self.graph.nodes[e1["v"]]["tag"], [])[0]:
                                self.graph.nodes[node]["side"] = "right"
                                side = "left"
                            else:
                                self.graph.nodes[node]["side"] = "left"
                                side = "right"


                    self.graph.add_node(
                        f"test_{p_end}", tag=p_end, kind="junction", subtype="junction", side=side, vert_side="top", source_degree="junction", base=self.graph.nodes[e1["v"]]["base"],
                        is_flange="is_flange", coords=coord
                    )

                    left_right_to_be_refeatured.append(f"test_{p_end}")

                    bottom_node = \
                    [node for node, data in self.graph.nodes(data=True) if data.get("tag") == bottom_fillet_tag][0]
                    to_be_refeatured.append(bottom_node)

                    if self.graph.nodes[bottom_node]["coords"][0] > self.graph.nodes[f"test_{p_end}"]["coords"][0]:
                        self.graph.nodes[f"test_{p_end}"]["side"] = "left"
                    else:
                        self.graph.nodes[f"test_{p_end}"]["side"] = "right"


                    right_line_tag = _find_bottom_fillet(e3["u"])
                    self.model.occ.synchronize()
                    _, pt_tags = self.model.getAdjacencies(1, right_line_tag)
                    bottom_fillet_tag = min(pt_tags, key=lambda pt: self.model.getValue(0, pt, [])[2])
                    corner = self.__add_corner(right_line_tag, e3["tag"], self.graph.nodes[e3["u"]]["tag"], self.r, adjust_radius=True, a_1_point=bottom_fillet_tag)
                    _, _, p_end = corner
                    self.model.occ.synchronize()

                    coord = self.model.getValue(0, p_end, [])

                    # check if there already exists a junction with the same base
                    for node, data in self.graph.nodes(data=True):
                        if data.get("kind") == "junction" and data.get("base") == self.graph.nodes[e1["u"]]["base"]:
                            if data.get("tag") == self.graph.nodes[e1["u"]]["tag"]:
                                continue
                            coordinates = data.get("coords")
                            # if new coordinates are >x than existing coordinates, reset the side to right
                            if coordinates[0] < self.model.getValue(0, self.graph.nodes[e1["u"]]["tag"], [])[0]:
                                self.graph.nodes[node]["side"] = "left"
                                side = "right"
                            else:
                                self.graph.nodes[node]["side"] = "right"
                                side = "left"

                    self.graph.add_node(
                        f"test_{p_end}", tag=p_end, kind="junction", subtype="junction", side=side, vert_side="top", source_degree="junction", base=self.graph.nodes[e3["u"]]["base"],
                        is_flange="is_flange", coords=coord
                    )

                    left_right_to_be_refeatured.append(f"test_{p_end}")

                    bottom_node = [node for node, data in self.graph.nodes(data=True) if data.get("tag") == bottom_fillet_tag][0]
                    to_be_refeatured.append(bottom_node)

                    if self.graph.nodes[bottom_node]["coords"][0] > self.graph.nodes[f"test_{p_end}"]["coords"][0]:
                        self.graph.nodes[f"test_{p_end}"]["side"] = "left"
                    else:
                        self.graph.nodes[f"test_{p_end}"]["side"] = "right"

        # group all candidate nodes by their base
        base_groups = defaultdict(list)
        for node in left_right_to_be_refeatured:
            base = self.graph.nodes[node]["base"]
            base_groups[base].append(node)

        # now go through each group and assign left/right
        for base, nodes in base_groups.items():
            if len(nodes) != 2:
                continue

            n1, n2 = nodes
            x1 = self.graph.nodes[n1]["coords"][0]
            x2 = self.graph.nodes[n2]["coords"][0]

            if x1 < x2:
                self.graph.nodes[n1]["side"] = "left"
                self.graph.nodes[n2]["side"] = "right"
            else:
                self.graph.nodes[n1]["side"] = "right"
                self.graph.nodes[n2]["side"] = "left"

        for node in set(to_be_refeatured):
            self.graph.nodes[node]["kind"] = "junction"
            self.graph.nodes[node]["subtype"] = "junction"
            self.graph.nodes[node]["side"] = "special"
            self.graph.nodes[node]["vert_side"] = None


    @final
    def _fillet_support_junctions(self) -> None:
        """
        At each endpoint of a 'rem' fillet (from degree-4 nodes),
        create a fillet between its original arc and the support edge
        introduced in connect_fillet_nodes_on_yz_planes().

        The new junction fillet endpoints are tagged with:
          - subtype="junction"
          - side="left"/"right" based on X coordinate
        """
        self.model.occ.synchronize()

        # Candidate nodes: endpoints of 'rem' fillets created at degree-4 nodes
        cand_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get("kind") == "fillet"
               and data.get("source_degree") == 4
               and data.get("subtype") == "rem"
        ]

        # Always form pairs [left, right]
        pairs = self.__make_left_right_pairs(cand_nodes)

        for left, right in pairs:
            for node in (left, right):
                pt_tag = self.graph.nodes[node]["tag"]
                if "left" in node:
                    side = "left"
                else:
                    side = "right"
                inc_edges = list(self.graph.edges(node, data=True))

                # classify edges
                arc_edges = [e for e in inc_edges if e[2].get("kind") == "arc"]
                support_edges = [e for e in inc_edges if e[2].get("kind") == "support"]

                if len(support_edges) != 2:
                    continue

                if not arc_edges or not support_edges:
                    continue  # Skip if nothing to fillet

                # Pick the "original corner arc": the one with "rem" in its name
                arc_e = next((e for e in arc_edges if "_rem" in e[2]["name"]), None)
                if arc_e is None:
                    continue

                # Fillet arc + support
                result = self.__add_corner(
                    edge_a_tag=support_edges[0][2]["tag"], edge_b_tag=support_edges[1][2]["tag"], corner_tag=pt_tag,
                    radius=self.r, adjust_radius=True
                )
                if not result:
                    continue

                self.model.occ.synchronize()
                arc_tag, p_start, p_end = result

                coords_start = tuple(gmsh.model.getValue(0, p_start, []))
                coords_end = tuple(gmsh.model.getValue(0, p_end, []))

                if coords_start[2] <= coords_end[2]:
                    vert_start, vert_end = "bottom", "top"
                else:
                    vert_start, vert_end = "top", "bottom"

                base = self.graph.nodes[node]["base"]

                n_start = self.__make_node_name(
                    base, "junction", "junction", side, vert_start,
                    is_flange=self.graph.nodes[node].get("is_flange")
                )
                n_end = self.__make_node_name(
                    base, "junction", "junction", side, vert_end,
                    is_flange=self.graph.nodes[node].get("is_flange")
                )

                self.graph.add_node(
                    n_start, tag=p_start, kind="junction", subtype="junction", side=side, vert_side=vert_start,
                    coords=coords_start, source_degree="junction", base=base,
                    is_flange=self.graph.nodes[node].get("is_flange")
                )
                self.graph.add_node(
                    n_end, tag=p_end, kind="junction", subtype="junction", side=side, vert_side=vert_end,
                    coords=coords_end, source_degree="junction", base=base,
                    is_flange=self.graph.nodes[node].get("is_flange")
                )

                self.graph.add_edge(n_start, n_end, name=f"arc_{node}_jf", tag=arc_tag, kind="arc", subtype="junction")

    @final
    def _debug_list_all_nodes(self) -> None:
        """ Print all nodes in the graph with their attributes. """
        for node, data in self.graph.nodes(data=True):
            print(
                f"{node}: "
                f"tag={data.get('tag')}, "
                f"kind={data.get('kind')}, "
                f"subtype={data.get('subtype')}, "
                f"side={data.get('side')}, "
                f"vert_side={data.get('vert_side')}, "
                f"base={data.get('base')}, "
                f"rename={data.get('rename')}, "
                f"is_flange={data.get('is_flange')}, "
            )

    @final
    def _debug_list_all_edges(self) -> None:
        """ Print all edges in the graph with their attributes. """
        for edge in self.graph.edges(data=True):
            print(
                f"{edge}: "
            )

    @final
    def _connect_all_junctions_with_bspline(self) -> None:
        """
        For every base node:
          - find junction nodes (left/right, top or bottom)
          - find the corresponding yz fillet node (top or bottom)
          - create a BSpline left -> yz -> right

        Creates two splines per base if both top and bottom exist.
        """
        combos = defaultdict(lambda: {"yz": [], "junctions": []})

        # Single pass over all nodes
        for node, data in self.graph.nodes(data=True):
            base, kind, is_flange = data.get("base"), data.get("kind"), data.get("is_flange")
            if not base:
                continue

            if kind == "fillet" and data.get("subtype") == "yz":
                combos[(base, is_flange, data.get("side"))]["yz"].append(node)

            elif kind == "junction" and data.get("subtype") == "junction" and data.get("side") in ("left", "right"):
                combos[(base, is_flange, data.get("vert_side"))]["junctions"].append(node)

        # Now iterate over prepared combos
        for (base, is_flange, vert), group in combos.items():
            yz_nodes = group["yz"]
            junction_nodes = group["junctions"]

            if not yz_nodes or len(junction_nodes) != 2:
                continue

            yz_node = yz_nodes[0]
            left, right = sorted(junction_nodes, key=lambda n: self.graph.nodes[n]["coords"][0])
            self.__create_bspline(base, vert, left, yz_node, right, is_flange)

    def _split_by_reference(self, nodes: list[str]) -> tuple[list[str], list[str]]:
        """
        Split nodes into 'below' and 'above' groups depending on class type.
        For UChannel1: use split_by_centerline().
        For others: use y_center.
        """
        if self.__class__.__name__ == "UChannel1":
            return split_by_centerline(
                self.graph,
                nodes,
                p1=(0, self.y_l / 2, 0),
                p2=((self.x_1 + self.x_2) / 2, self.y_r / 2, 0)
            )
        else:
            if self.bounding_box[1] < self.bounding_box[4]:
                y_center = (self.bounding_box[1] + self.bounding_box[4]) / 2
            else:
                y_center = (self.bounding_box[4] + self.bounding_box[1]) / 2

            below = [n for n in nodes if self.graph.nodes[n]["coords"][1] <= y_center]
            above = [n for n in nodes if self.graph.nodes[n]["coords"][1] >= y_center]
            return below, above

    def __create_bspline(self, base: int, vert: str, left: str, mid: str, right: str, is_flange: Optional[str]) -> None:
        """ Create a BSpline left -> mid -> right and add it to the graph. """
        p_left = self.graph.nodes[left]["tag"]
        p_mid = self.graph.nodes[mid]["tag"]
        p_right = self.graph.nodes[right]["tag"]

        tag = self.model.occ.addSpline([p_left, p_mid, p_right])

        self.graph.add_edge(
            left, right, name=f"bspline_{base}_{vert}", tag=tag, kind="bspline", subtype=vert, mid=mid,
            is_flange=is_flange
        )

    @final
    def _connect_chain(self, vert_side: str) -> list[tuple[str, str, int, str, str]]:
        """
        Build chains along one vertical side ('top' or 'bottom') in 2 passes:
          Pass 1: fillets below y_ref
          Pass 2: fillets above y_ref
        Ensures the two passes do not reuse the same start/end nodes.

        :param vert_side: 'top' or 'bottom'
        :return: list of (start, end, tag, pass_side, step_side)
              - pass_side: "below" or "above"
              - step_side: "left", "right", or "direct"
        """
        all_connections = []
        for is_flange in ["is_flange", None]:
            # Collect eligible fillet nodes
            all_fillets = [
                node for node, data in self.graph.nodes(data=True)
                if data.get("kind") == "fillet"
                   and data.get("subtype") == "yz"
                   and data.get("vert_side") is None
                   and data.get("side") == vert_side
                   and data.get("is_flange") is is_flange
            ]
            if not all_fillets:
                continue

            # Get base nodes
            base_to_nodes = {}
            for node, data in self.graph.nodes(data=True):
                base_to_nodes.setdefault(data.get("base"), []).append((node, data))

            below, above = self._split_by_reference(all_fillets)

            if len(below) < 2 or len(above) < 2:
                return all_connections

            # First pass (below)
            start_below = min(below, key=lambda n: self.graph.nodes[n]["coords"][0])
            end_below = max(below, key=lambda n: self.graph.nodes[n]["coords"][0])
            conns = self.__walk_chain(vert_side, start_below, end_below,
                                      is_flange, pass_side="below")
            all_connections.extend(conns)

            # Second pass (above y center)
            end_above = max(above, key=lambda n: (self.graph.nodes[n]["coords"][0]))
            start_above = min(above, key=lambda n: (self.graph.nodes[n]["coords"][0]))

            # Avoid reusing start/end from first pass
            if {start_above, end_above} & {start_below, end_below}:
                remaining = [n for n in above if n not in (start_below, end_below)]
                if len(remaining) >= 2:
                    start_above = min(remaining, key=lambda n: (self.graph.nodes[n]["coords"][0]))
                    end_above = max(remaining, key=lambda n: (self.graph.nodes[n]["coords"][0]))
                else:
                    return all_connections

            conns = self.__walk_chain(vert_side, start_above, end_above,
                                      is_flange, pass_side="above")
            all_connections.extend(conns)

        return all_connections

    def __walk_chain(self, vert_side: str, start: str, end: str, is_flange, pass_side: str) -> List[Tuple[str, str, int, str, str]]:
        """
        From a start node, walk along the x-axis to find the next junction till we reach the end node.
        :param vert_side: 'top' or 'bottom'
        :param start: start node
        :param end: end node
        :param is_flange: True if the chain is a flange
        :param pass_side: "below" or "above"
        :return: list of connections
        """
        visited = {start}
        current = start
        chain = [start]
        connections = []

        while True:
            best_left = self.__find_next_left_junction(
                current, end, visited, vert_side, is_flange, pass_side
            )
            if best_left is None:
                break

            side = self.graph.nodes[best_left].get("side")

            if self.ad and len(visited) == 1 and self.graph.nodes[current]["is_flange"] == None:
                if pass_side == "above":
                    a = self.model.occ.addPoint(
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[0] + 0.25 * self.x_ad,
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[1],
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[2] - 0.1 * self.z_ad)
                    b = self.model.occ.addPoint(
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[0] + 0.50 * self.x_ad,
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[1],
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[2] - 0.5 * self.z_ad)
                    c = self.model.occ.addPoint(
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[0] + 0.75 * self.x_ad,
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[1],
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[2] - 0.9 * self.z_ad)
                    tag = self.model.occ.addSpline([
                        self.graph.nodes[current]["tag"],
                        a,
                        b,
                        c,
                        self.graph.nodes[best_left]["tag"]
                    ])
                    self.model.occ.remove([(0, a), (0, b), (0, c)], recursive=True)
                elif pass_side == "below":
                    a = self.model.occ.addPoint(
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[0] + 0.25 * self.x_ad,
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[1],
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[2] - 0.1 * self.z_ad)
                    b = self.model.occ.addPoint(
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[0] + 0.50 * self.x_ad,
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[1],
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[2] - 0.5 * self.z_ad)
                    c = self.model.occ.addPoint(
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[0] + 0.75 * self.x_ad,
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[1],
                        self.model.getValue(0, self.graph.nodes[current]["tag"], [])[2] - 0.9 * self.z_ad)
                    tag = self.model.occ.addSpline([
                        self.graph.nodes[current]["tag"],
                        a,
                        b,
                        c,
                        self.graph.nodes[best_left]["tag"]
                    ])
                    self.model.occ.remove([(0, a), (0, b), (0, c)], recursive=True)
                else:
                    continue
            else:

                tag = self.model.occ.addLine(
                    self.graph.nodes[current]["tag"],
                    self.graph.nodes[best_left]["tag"]
                )

            self.graph.add_edge(
                current, best_left,
                name=f"chain_{current}_{best_left}",
                tag=tag, kind="chain", subtype=vert_side
            )

            connections.append((current, best_left, tag, pass_side, "left"))
            visited.add(best_left)
            chain.append(best_left)

            if side == "special":
                # Special nodes don’t require a right partner
                current = best_left
                continue

            partner = self.__find_partner_right_junction(best_left, visited, vert_side)
            if partner is None:
                break

            visited.add(partner)
            chain.append(partner)
            current = partner

        if current != end:
            tag = self.model.occ.addLine(
                self.graph.nodes[current]["tag"],
                self.graph.nodes[end]["tag"]
            )

            self.graph.add_edge(
                current, end,
                name=f"chain_{current}_{end}",
                tag=tag, kind="chain", subtype=vert_side
            )

            step_side = "direct" if not connections else "right"
            connections.append((current, end, tag, pass_side, step_side))

            visited.add(end)
            chain.append(end)

        return connections

    def __find_next_left_junction(self, current: str, end: str, visited: set[str] | None, vert_side: str, is_flange, pass_side: str) -> str | None:
        """Find next left junction for given vert_side ('top' or 'bottom')."""
        if visited is None:
            visited = set()

        graph = self.graph
        if current not in graph or end not in graph:
            return None

        sx, *_, sz = graph.nodes[current]["coords"]
        ex, ey, *_ = graph.nodes[end]["coords"]

        if vert_side == "top" and is_flange == "is_flange":
            candidates = [
                node for node, data in graph.nodes(data=True)
                if (
                    (
                        # --- existing junction condition ---
                            data.get("kind") == "junction"
                            and data.get("subtype") == "junction"
                            and data.get("is_flange") == is_flange
                            and node not in visited
                            and (
                                # condition A
                                    (
                                            (data.get("side") in ["left", "special"])
                                            and data.get("vert_side") == vert_side
                                    )
                            )
                    )
                )
            ]

        else:

            candidates = [
                node for node, data in graph.nodes(data=True)
                if (
                        (
                            # --- existing junction condition ---
                                data.get("kind") == "junction"
                                and data.get("subtype") == "junction"
                                and data.get("is_flange") == is_flange
                                and node not in visited
                                and (
                                    # condition A
                                        (
                                                (data.get("side") in ["left", "special"])
                                                and data.get("vert_side") == vert_side
                                        )
                                        # OR condition B
                                        or (
                                                data.get("side") == "special"
                                                and data.get("vert_side") is None
                                        )
                                )
                        )
                )
            ]

        if not candidates:
            return None

            # Keep only those in the same group as current, based on pass_side
        below, above = self._split_by_reference([current] + candidates)
        group = below if pass_side == "below" else above
        candidates = [n for n in candidates if n in group]

        if not candidates:
            return None

        cx = graph.nodes[current]["coords"][0]
        ex = graph.nodes[end]["coords"][0]
        direction = 1 if ex > cx else -1

        candidates = [
            n for n in candidates
            if direction * (graph.nodes[n]["coords"][0] - cx) > 0
        ]

        if not candidates:
            return None

        best = min(
            candidates,
            key=lambda n: (
                    (graph.nodes[n]["coords"][0] - sx) ** 2 +
                    (graph.nodes[n]["coords"][2] - sz) ** 2
            )
        )

        bx, *_ = graph.nodes[best]["coords"]

        if abs(ex - sx) < abs(bx - sx):
            return None

        return best

    def __find_partner_right_junction(self, left_node: str, visited: set[str] | None, vert_side: str) -> str | None:
        """Find right partner junction for the same base and given vert_side ('top' or 'bottom')."""
        if visited is None:
            visited = set()

        graph = self.graph
        if left_node not in graph:
            return None

        base = graph.nodes[left_node]["base"]

        partner = next(
            (
                node
                for node, data in graph.nodes(data=True)
                if (
                    data.get("base") == base
                    and node not in visited
                    and data.get("kind") == "junction"
                    and data.get("subtype") == "junction"
                    and data.get("side") == "right"
                    and data.get("vert_side") == vert_side
            )
            ),
            None,
        )
        return partner

    def __edges_in_yz_plane(self, node: str, tolerance: float = 1e-8):
        """Return edges connected to `node` whose neighbor has same x (YZ-plane)."""
        x0 = self.graph.nodes[node]["coords"][0]
        yz_edges = []
        for u, v, data in self.graph.edges(node, data=True):
            other = v if u == node else u
            x_other = self.graph.nodes[other]["coords"][0]
            if abs(x_other - x0) < tolerance:
                yz_edges.append((u, v, data))
        return yz_edges

    def __get_most_collinear_pair(self, node: str, edges: List) -> tuple:
        """From edges at a node, return the pair that is most collinear."""
        best = None
        best_score = -1.0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                v1 = self.__edge_vector(node, edges[i])
                v2 = self.__edge_vector(node, edges[j])
                score = abs(np.dot(v1, v2))  # 1 = perfectly collinear
                if score > best_score:
                    best_score = score
                    best = (edges[i], edges[j])
        return best

    def __make_left_right_pairs(self, candidates: list[str]) -> list[list[str]]:
        """
        Given candidate nodes, return a list of [left, right] pairs.
        Each pair has exactly one node with side="left" and one with side="right".
        Pairs are formed by ordering each side by closeness (x,y,z distance to origin).
        """
        left, right = [], []

        for node in candidates:
            data = self.graph.nodes[node]
            side = data.get("side")
            if side not in ("left", "right"):
                continue

            x, y, z = data["coords"]
            dist2 = x * x + y * y + z * z  # squared distance to origin
            if side == "left":
                left.append((dist2, node))
            else:  # side == "right"
                right.append((dist2, node))

        left.sort(key=lambda t: t[0])
        right.sort(key=lambda t: t[0])

        # Pair them up: always [left, right]
        pairs = []
        for l, r in zip(left, right):
            pairs.append([l[1], r[1]])

        return pairs

    @final
    def _rebuild_graph_from_occ(self) -> None:
        """Rebuild self.graph from current OCC curves (dim=1)."""
        self.graph = nx.Graph() # Redefine clean empty graph

        # Collect all points first
        points = gmsh.model.getEntities(0)
        for _, pt in points:
            coords = tuple(gmsh.model.getValue(0, pt, []))
            self.graph.add_node(f"p{pt}", tag=pt, coords=coords, kind="point")

        # Collect all curves
        for _, curve_tag in gmsh.model.getEntities(1):
            dim, adj = gmsh.model.getAdjacencies(1, curve_tag)  # (dim, [point tags])
            if len(adj) != 2:
                continue
            p1, p2 = adj
            n1, n2 = f"p{p1}", f"p{p2}"
            self.graph.add_edge(n1, n2, tag=curve_tag, kind="line")

    @final
    def _clean_graph(self) -> None:
        """Keep only the largest connected component, remove dangling edges,
        and also remove corresponding entities from OCC."""
        if self.graph.number_of_nodes() == 0:
            return

        # Step 1: Find connected components
        components = list(nx.connected_components(self.graph))
        if not components:
            return

        largest_component = max(components, key=len)

        # Step 2: Remove all nodes/edges NOT in the largest component from OCC
        for comp in components:
            if comp is largest_component:
                continue  # skip the big one
            for node in comp:
                # remove the OCC point
                self.model.occ.remove([(0, self.graph.nodes[node]["tag"])], recursive=True)
                # remove its OCC edges
                for _, nbr, data in self.graph.edges(node, data=True):
                    self.model.occ.remove([(1, data["tag"])], recursive=True)

        # Step 3: Work only with the largest component in networkx
        keep_graph = self.graph.subgraph(largest_component).copy()

        # Step 4: Iteratively remove dangling nodes + corresponding OCC edges
        while True:
            dangling_nodes = [n for n, d in keep_graph.degree() if d == 1]
            if not dangling_nodes:
                break

            occ_to_remove = []
            for node in dangling_nodes:
                occ_to_remove.append((0, keep_graph.nodes[node]["tag"]))  # remove point
                for _, nbr, data in keep_graph.edges(node, data=True):
                    occ_to_remove.append((1, data["tag"]))  # remove edge

            if occ_to_remove:
                self.model.occ.remove(occ_to_remove, recursive=True)

            keep_graph.remove_nodes_from(dangling_nodes)

        # Replace the old graph
        self.graph = keep_graph

    def __signed_tag_between(self, n1: str, n2: str) -> int | None:
        """
        Return the oriented curve tag for the edge (n1 -> n2):
          +tag if OCC endpoints are (n1, n2)
          -tag if OCC endpoints are (n2, n1)
        """
        edge_data = self.graph.get_edge_data(n1, n2)
        if not edge_data:
            return None
        tag = edge_data.get("tag")
        if tag is None:
            return None

        # OCC endpoints in curve orientation
        _, pts = gmsh.model.getAdjacencies(1, tag)
        if len(pts) != 2:
            return None

        p_n1 = self.graph.nodes[n1]["tag"]  # gmsh point tag stored on node
        p_n2 = self.graph.nodes[n2]["tag"]

        if pts[0] == p_n1 and pts[1] == p_n2:
            return tag
        if pts[0] == p_n2 and pts[1] == p_n1:
            return -tag

        # Fallback: keep positive (should not generally happen if graph matches OCC)
        return tag

    def __find_4_cycles_ordered(self) -> list[list[int]]:
        """
        Find all 4-cycles using common neighbors, and return **ordered, oriented**
        curve tag lists suitable for gmsh.addCurveLoop.
        Deduplicates by set of absolute tags.
        """
        G = self.graph
        nodes = list(G.nodes())
        loops: list[list[int]] = []
        seen: set[frozenset[int]] = set()

        for i in range(len(nodes)):
            u = nodes[i]
            for j in range(i + 1, len(nodes)):
                v = nodes[j]

                # common neighbors of u and v
                common = list(nx.common_neighbors(G, u, v))
                if len(common) < 2:
                    continue

                for x, y in itertools.combinations(common, 2):
                    # two possible 4-cycles: u-x-v-y-u and u-y-v-x-u
                    for order in ((u, x, v, y), (u, y, v, x)):
                        n0, n1, n2, n3 = order

                        # make sure the 4 required edges exist
                        if not (G.has_edge(n0, n1) and G.has_edge(n1, n2)
                                and G.has_edge(n2, n3) and G.has_edge(n3, n0)):
                            continue

                        # build **signed** curve list following the node order
                        s01 = self.__signed_tag_between(n0, n1)
                        s12 = self.__signed_tag_between(n1, n2)
                        s23 = self.__signed_tag_between(n2, n3)
                        s30 = self.__signed_tag_between(n3, n0)
                        if None in (s01, s12, s23, s30):
                            continue

                        signed = [s01, s12, s23, s30]

                        # dedupe by **absolute** tag set (orientation/rotation agnostic)
                        key = frozenset(abs(t) for t in signed)
                        if key in seen:
                            continue
                        seen.add(key)
                        loops.append(signed)

        return loops

    def __find_3_cycles_ordered(self) -> list[list[int]]:
        """
        Find all 3-cycles (triangular holes) and return ordered, oriented
        curve tag lists suitable for gmsh.addCurveLoop.
        Deduplicates by set of absolute tags.
        """
        G = self.graph
        nodes = list(G.nodes())
        loops: list[list[int]] = []
        seen: set[frozenset[int]] = set()

        for i in range(len(nodes)):
            u = nodes[i]
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                for k in range(j + 1, len(nodes)):
                    w = nodes[k]

                    # must form a triangle
                    if not (G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(w, u)):
                        continue

                    # build signed curve tags
                    s_uv = self.__signed_tag_between(u, v)
                    s_vw = self.__signed_tag_between(v, w)
                    s_wu = self.__signed_tag_between(w, u)
                    if None in (s_uv, s_vw, s_wu):
                        continue

                    signed = [s_uv, s_vw, s_wu]

                    # dedupe by absolute tags
                    key = frozenset(abs(t) for t in signed)
                    if key in seen:
                        continue
                    seen.add(key)
                    loops.append(signed)

        return loops

    @final
    def _build_surfaces_from_ordered_cycles(self) -> List[int]:
        """ Use ordered 3- and 4-cycles to create BSpline surfaces. """
        self.model.occ.synchronize()

        loops = []
        loops.extend(self.__find_4_cycles_ordered())
        loops.extend(self.__find_3_cycles_ordered())

        surfaces = []
        for signed_tags in loops:
            try:
                loop_tag = self.model.occ.addCurveLoop(signed_tags)
                surf_tag = self.model.occ.addSurfaceFilling(loop_tag, degree=2, maxDegree=3)
                surfaces.append(surf_tag)
            except Exception as e:
                print(f"[surfaces] ! failed from {signed_tags}: {e}")

        self.model.occ.synchronize()
        return surfaces

    def _build_shell_from_surfaces(self, surface_tags: List[int]):
        """
        Given a list of surface tags, stitch them into a shell.
        Returns the shell loop tag.
        """
        if not surface_tags:
            return None

        sf_tags = [tag for dim, tag in gmsh.model.occ.getEntities(2)]
        sl_tag = gmsh.model.occ.addSurfaceLoop(sf_tags, 1)
        v_tag = gmsh.model.occ.addVolume([sl_tag], 1)
        self.model.occ.synchronize()
        return v_tag

    @staticmethod
    def __make_node_name(base: str | int, kind: str, subtype: str, side: str, vert: Optional[str] = None, is_flange: Optional[str] = None) -> str:
        """ Create a node name based on the given parameters. This is a helper function for the graph creation. """
        if vert:
            if is_flange:
                return f"{base}_{kind}_{subtype}_{side}_{vert}_{is_flange}"
            return f"{base}_{kind}_{subtype}_{side}_{vert}_{is_flange}"
        if is_flange:
            return f"{base}_{kind}_{subtype}_{side}_{is_flange}"
        return f"{base}_{kind}_{subtype}_{side}"

    def get_segmentation_labels(self, has_fillet: Optional[bool] = None, has_flange: Optional[bool] = None) -> dict:
        """
        Compute segmentation labels using the FaceSegmenter.
        :param has_fillet: Optional override for fillet detection.
        :param has_flange: Optional override for flange detection.
        :return: Dict of classified face labels.
        """
        # TODO: Segmentation doesnt seem to work 100% reliable. Revisit this.
        if not hasattr(self, "graph") or self.graph is None:
            raise ValueError("Face adjacency graph must be created before segmentation.")

        # Detect fillet condition
        if has_fillet is None:
            if hasattr(self, "r_l") and hasattr(self, "r_r"):  # UChannel1 style
                has_fillet = any(r > 0 for r in (*self.r_l, *self.r_r))
            elif hasattr(self, "r"):  # UChannel2/3/4 style
                has_fillet = self.r > 0
            else:
                has_fillet = False

        # Detect flange condition
        if has_flange is None:
            has_flange = getattr(self, "fl", False)

        self.face_segmenter = FaceSegmenter(self.graph, has_fillet_radii=has_fillet, has_flange=has_flange)
        self.segmentation_labels = self.face_segmenter.segment_faces()
        return self.segmentation_labels

    def visualize_segmentation_graph(self) -> None:
        """ Plot segmented face nodes in 3D. """
        if not self.segmentation_labels:
            raise ValueError("There are no segmentation labels. segmentation_label = True in self.create_geometry?")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Pull positions from node attribute 'center'
        positions = {node: data.get("center", (0, 0, 0)) for node, data in self.graph.nodes(data=True)}

        class_colors = {
            "bottom_faces": "dodgerblue",
            "fillet_die_faces": "limegreen",
            "flange_faces": "darkorange",
            "slant_faces": "purple",
            "fillet_punch_faces": "crimson",
            "start_node": "black"
        }

        # Draw classified nodes
        for class_name, nodes in self.segmentation_labels.items():
            if isinstance(nodes, int):
                nodes = [nodes]

            color = class_colors.get(class_name, "gray")
            for node in nodes:
                pos = positions.get(node)
                if pos is not None:
                    ax.scatter(*pos, s=100, color=color, depthshade=False)

        # Draw edges
        for u, v in self.graph.edges():
            u_pos, v_pos = positions.get(u), positions.get(v)
            if u_pos is not None and v_pos is not None:
                ax.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], [u_pos[2], v_pos[2]], color='gray')

        # Add legend
        legend_elements = []
        for label, color in class_colors.items():
            nodes = self.segmentation_labels.get(label)
            if nodes:
                # Convert to list if it's a single int
                if isinstance(nodes, int):
                    nodes = [nodes]
                if len(nodes) > 0:
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', label=label.replace("_", " ").title(),
                               markerfacecolor=color, markersize=10)
                    )

        ax.legend(handles=legend_elements, loc='upper right', title='Face Segmentation Labels')

        set_axes_equal(ax)

        plt.show()

    @staticmethod
    def create_image(input_path: Path, output_path: Path, pov="isometric x", title="", zoom_factor=1.0, curve_width=2.0,
                     show=False):
        """
        Create an image of a geometry using gmsh. The background color is set to white.

        :param input_path: path to the geometry
        :param output_path: path to the exported image
        :param pov: point of view (analogous to LS-PrePost)
        :param title: Text to print into the top of the image
        :param zoom_factor: zoom_factor > 1.0: zoom in
                            zoom_factor < 1.0: zoom out
        :param curve_width: factor to adapt the width of the edges in the image. THe larger the value, the thicker the
                            edges.
        :param show: Option to show the image in the gui
        """
        # load geometry
        gmsh.initialize()
        gmsh.model.add("")
        gmsh.merge(str(input_path))

        # set background color
        white = (255, 255, 255)
        black = (0, 0, 0)
        gmsh.option.setColor("General.Background", white[0], white[1], white[2])
        gmsh.option.setColor("General.Foreground", black[0], black[1], black[2])
        gmsh.option.setColor("General.Text", black[0], black[1], black[2])

        # rotate to given perspective
        if pov == "front":
            x_rot = -90
            y_rot = 0
            z_rot = 90
        elif pov == "back":
            x_rot = -90
            y_rot = 0
            z_rot = 270
        elif pov == "left":
            x_rot = -90
            y_rot = 0
            z_rot = 0
        elif pov == "right":
            x_rot = -90
            y_rot = 0
            z_rot = 180
        elif pov == "top":
            x_rot = 0
            y_rot = 0
            z_rot = 0
        elif pov == "bottom":
            x_rot = 0
            y_rot = 0
            z_rot = 180
        elif pov == "isometric x":
            x_rot = -45
            y_rot = 0
            z_rot = 45 + 180
        elif pov == "isometric y":
            x_rot = 45
            y_rot = 45
            z_rot = 90
        elif pov == "isometric z":
            x_rot = 45
            y_rot = -45
            z_rot = 0
        elif pov == "isometric -x":
            x_rot = -45
            y_rot = 0
            z_rot = 45
        elif pov == "isometric -y":
            x_rot = 45
            y_rot = 45 + 180
            z_rot = 90
        elif pov == "isometric -z":
            x_rot = 45
            y_rot = -45 + 180
            z_rot = 0
        else:
            raise ValueError(f'There exists no point of view: {pov}. Please adapt your input.')

        gmsh.option.setNumber("General.Trackball", 0)
        gmsh.option.setNumber("General.RotationX", x_rot)
        gmsh.option.setNumber("General.RotationY", y_rot)
        gmsh.option.setNumber("General.RotationZ", z_rot)

        # zoom
        gmsh.option.setNumber("General.ScaleX", zoom_factor)
        gmsh.option.setNumber("General.ScaleY", zoom_factor)
        gmsh.option.setNumber("General.ScaleZ", zoom_factor)

        # adapt the width of the edges
        gmsh.option.setNumber("Geometry.CurveWidth", curve_width)

        # show the gui or run in background
        if show:
            gmsh.fltk.run()
        else:
            gmsh.fltk.initialize()

        # title
        if title != "":
            gmsh.view.add("my_view", 1)  # Set the view index
            gmsh.plugin.setString("Annotate", "Text", title)
            gmsh.plugin.setNumber("Annotate", "X", 1.e5)
            gmsh.plugin.setNumber("Annotate", "Y", 50)
            gmsh.plugin.setString("Annotate", "Font", "Times-BoldItalic")
            gmsh.plugin.setNumber("Annotate", "FontSize", 28)
            gmsh.plugin.setString("Annotate", "Align", "Center")
            gmsh.plugin.run("Annotate")

        # export and stop gmsh
        gmsh.write(str(output_path))
        gmsh.fltk.finalize()
        gmsh.finalize()


if __name__ == '__main__':
    pass
