from pathlib import Path
from typing import Tuple
import gmsh
import math
import networkx as nx

from uchannel.uchannel_base import UChannelBase
from uchannel.error_handling import FeasibilityError


class UChannel1(UChannelBase):
    def __init__(self, path: Path, label_path: Path, x_1: float = 100, x_2: float = 105, y_l: float = 50, y_r: float = 50,
                 y_fl: float = 50, z_l_1: float = 33.3, z_l_2: float = 33.3, z_r_1: float = 33.3, z_r_2: float = 33.3,
                 o_mp_1: Tuple[float, float, float] = (1, 1, 1),
                 o_mp_2: Tuple[float, float, float] = (1, 1, 1), r_l: Tuple[float, float] = (5, 5),
                 r_r: Tuple[float, float] = (5, 5), beta_l_1: float = 15, beta_l_2: float = 15, beta_r_1: float = 15,
                 beta_r_2: float = 15, fl: bool = 15):
        """
        Class for generating the UChannel1 geometry in gmsh. A short description of the parameters is given below.

        The key parameter for feasibility control is x_1. It can take any value. The authors recommend values between
        274 and 2477 mm. The remaining parameters are limited by x_1.

        :param path: Path to the .step file where the geometry will be exported
        :param label_path: Path to the .json file where the segmentation labels will be exported
        :param x_1: Length of the bottom plane on side 1 along the x-axis
        :param x_2: Length of the bottom plane on side 2 along the x-axis
        :param y_l: Width of the bottom plane on left side along the y-axis
        :param y_r: Width of the bottom plane on right side along the y-axis
        :param y_fl: Width of the flange along the y-axis
        :param z_l_1: Height of the left slant on side 1 along the z-axis
        :param z_l_2: Height of the left slant on side 2 along the z-axis
        :param z_r_1: Height of the right slant on side 1 along the z-axis
        :param z_r_2: Height of the right slant on side 2 along the z-axis
        :param o_mp_1: Offset of the mid-point of the bottom plane (side 1) along the x, y, and z-axis
        :param o_mp_2: Offset of the mid-point of the bottom plane (side 1) along the x, y, and z-axis
        :param r_l: Radii of the fillets on the left side
        :param r_r: Radii of the fillets on the right side
        :param beta_l_1: Angle of the left slant on side 1
        :param beta_l_2: Angle of the left slant on side 2
        :param beta_r_1: Angle of the right slant on side 1
        :param beta_r_2: Angle of the right slant on side 2
        :param fl: Boolean to indicate if the flange should be created
        """
        super().__init__(path, label_path)

        # Parameters
        self.x_1 = x_1
        self.x_2 = x_2
        self.y_l = y_l
        self.y_r = y_r
        self.y_fl = y_fl
        self.z_l_1 = z_l_1
        self.z_l_2 = z_l_2
        self.z_r_1 = z_r_1
        self.z_r_2 = z_r_2
        self.o_mp_1 = self.Vertex(*o_mp_1)
        self.o_mp_2 = self.Vertex(*o_mp_2)
        self.r_l = r_l
        self.r_r = r_r
        self.beta_l_1 = math.radians(beta_l_1)
        self.beta_l_2 = math.radians(beta_l_2)
        self.beta_r_1 = math.radians(beta_r_1)
        self.beta_r_2 = math.radians(beta_r_2)
        self.fl = fl

        self._calculate_vertices()
        self.f_bounds = self._feasibility_bounds()  # Feasibility bounds

    def _calculate_vertices(self) -> None:
        """
        Calculate the vertices of the UChannel1 geometry based on the parameters given in the constructor of UChannel1.
        The vertices order can be seen in the provided sketch.
        """
        # Vertices that construct the bottom plane
        self.v1 = self.Vertex(0, self.y_l, 0)
        self.v2 = self.Vertex(self.x_1, self.y_r, 0)
        self.v3 = self.Vertex(0, 0, 0)
        self.v4 = self.Vertex(self.x_2, 0, 0)
        self.v5 = self.Vertex((self.v2.x + self.v1.x) / 2 + self.o_mp_1.x,
                              (self.v2.y + self.v1.y) / 2 + self.o_mp_1.y, 0)
        self.v6 = self.Vertex((self.v3.x + self.v4.x) / 2 + self.o_mp_2.x,
                              (self.v3.y + self.v4.y) / 2 + self.o_mp_2.y, 0)

        # Vertices that construct the slants 1 and 2
        self.v7 = self.Vertex(0, self.y_l + self.z_l_1 * math.tan(self.beta_l_1), self.z_l_1)
        self.v8 = self.Vertex(self.x_1, self.y_r + self.z_r_1 * math.tan(self.beta_r_1), self.z_r_1)
        self.v9 = self.Vertex(0, -self.z_l_2 * math.tan(self.beta_l_2), self.z_l_2)
        self.v10 = self.Vertex(self.x_2, -self.z_r_2 * math.tan(self.beta_r_2), self.z_r_2)
        self.v11 = self.Vertex((self.v8.x + self.v7.x) / 2 + self.o_mp_1.x,
                               (self.v8.y + self.v7.y) / 2 + self.o_mp_1.y,
                               (self.v8.z + self.v7.z) / 2 + self.o_mp_1.z)
        self.v12 = self.Vertex((self.v10.x + self.v9.x) / 2 + self.o_mp_2.x,
                               (self.v10.y + self.v9.y) / 2 + self.o_mp_2.y,
                               (self.v10.z + self.v9.z) / 2 + self.o_mp_2.z)

        # Vertices that construct the flange 1 and 2
        if self.fl:
            self.v13 = self.Vertex(self.v7.x, self.v7.y + self.y_fl, self.v7.z)
            self.v14 = self.Vertex(self.v8.x, self.v8.y + self.y_fl, self.v8.z)
            self.v15 = self.Vertex(self.v9.x, self.v9.y - self.y_fl, self.v9.z)
            self.v16 = self.Vertex(self.v10.x, self.v10.y - self.y_fl, self.v10.z)
            self.v17 = self.Vertex(self.v11.x, self.v11.y + self.y_fl, self.v11.z)
            self.v18 = self.Vertex(self.v12.x, self.v12.y - self.y_fl, self.v12.z)

    def _feasibility_bounds(self) -> dict:
        """
        Bounds for the feasibility check. Can be used as an input for an LHS sampler.
        """
        bounds = {
            "x_2": self.Bounds(0.95 * self.x_1, 1.05 * self.x_1),
            "y_l": self.Bounds(0.1 * self.x_1, 1 / 2 * self.x_1),
            "y_r": self.Bounds(0.1 * self.x_1, 1 / 2 * self.x_1),
            "y_fl": self.Bounds(1 / 40 * self.x_1, 1 / 2 * self.x_1),
            "z_l_1": self.Bounds(2 / 20 * self.x_1, 1 / 3 * self.x_1),
            "z_l_2": self.Bounds(2 / 20 * self.x_1, 1 / 3 * self.x_1),
            "z_r_1": self.Bounds(2 / 20 * self.x_1, 1 / 3 * self.x_1),
            "z_r_2": self.Bounds(2 / 20 * self.x_1, 1 / 3 * self.x_1),
            "o_mp_1_x": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "o_mp_1_y": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "o_mp_1_z": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "o_mp_2_x": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "o_mp_2_y": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "o_mp_2_z": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "r_l_0": self.Bounds(2 / 1000 * self.x_1, 1 / 20 * self.x_1),
            "r_l_1": self.Bounds(2 / 1000 * self.x_1, 1 / 20 * self.x_1),
            "r_r_0": self.Bounds(2 / 1000 * self.x_1, 1 / 20 * self.x_1),
            "r_r_1": self.Bounds(2 / 1000 * self.x_1, 1 / 20 * self.x_1),
            "beta_l_1": self.Bounds(math.radians(5), math.radians(15)),
            "beta_l_2": self.Bounds(math.radians(5), math.radians(15)),
            "beta_r_1": self.Bounds(math.radians(5), math.radians(15)),
            "beta_r_2": self.Bounds(math.radians(5), math.radians(15)),
            "fl": self.Bounds(True, False)
        }
        return bounds

    def check_feasibility(self) -> bool:
        """
        Check if the parameters are feasible for geometry generation.
        The idea is that x_m can be any value, but the other parameters are limited by the value of x_1.

        :return: True if the parameters are feasible, False otherwise
        """
        try:
            # Validate individual parameters against their constraints
            # Checks for realistic proportions
            self._validate_parameter("x_2", self.x_2, self.f_bounds["x_2"].min, self.f_bounds["x_2"].max)
            self._validate_parameter("y_l", self.y_l, self.f_bounds["y_l"].min, self.f_bounds["y_l"].max)
            self._validate_parameter("y_r", self.y_r, self.f_bounds["y_r"].min, self.f_bounds["y_r"].max)
            self._validate_parameter("y_fl", self.y_fl, self.f_bounds["y_fl"].min, self.f_bounds["y_fl"].max)
            self._validate_parameter("z_l_1", self.z_l_1, self.f_bounds["z_l_1"].min, self.f_bounds["z_l_1"].max)
            self._validate_parameter("z_l_2", self.z_l_2, self.f_bounds["z_l_2"].min, self.f_bounds["z_l_2"].max)
            self._validate_parameter("z_r_1", self.z_r_1, self.f_bounds["z_r_1"].min, self.f_bounds["z_r_1"].max)
            self._validate_parameter("z_r_2", self.z_r_2, self.f_bounds["z_r_2"].min, self.f_bounds["z_r_2"].max)
            self._validate_parameter("o_mp_1[0]", self.o_mp_1.x, self.f_bounds["o_mp_1_x"].min, self.f_bounds["o_mp_1_x"].max)
            self._validate_parameter("o_mp_1[1]", self.o_mp_1.y, self.f_bounds["o_mp_1_y"].min, self.f_bounds["o_mp_1_y"].max)
            self._validate_parameter("o_mp_1[2]", self.o_mp_1.z, self.f_bounds["o_mp_1_z"].min, self.f_bounds["o_mp_1_z"].max)
            self._validate_parameter("o_mp_2[0]", self.o_mp_2.x, self.f_bounds["o_mp_2_x"].min, self.f_bounds["o_mp_2_x"].max)
            self._validate_parameter("o_mp_2[1]", self.o_mp_2.y, self.f_bounds["o_mp_2_y"].min, self.f_bounds["o_mp_2_y"].max)
            self._validate_parameter("o_mp_2[2]", self.o_mp_2.z, self.f_bounds["o_mp_2_z"].min, self.f_bounds["o_mp_2_z"].max)
            self._validate_parameter("y_fl", self.y_fl, self.f_bounds["y_fl"].min, self.f_bounds["y_fl"].max)
            self._validate_parameter("y_fl", self.y_fl, self.f_bounds["y_fl"].min, self.f_bounds["y_fl"].max)

            # Checks for geometry generation constraints (Overlaps, Intersections, Gmsh Feasibility etc.)
            self._validate_parameter("r_l[0]", self.r_l[0], self.f_bounds["r_l_0"].min, self.f_bounds["r_l_0"].max)
            self._validate_parameter("r_l[1]", self.r_l[1], self.f_bounds["r_l_1"].min, self.f_bounds["r_l_1"].max)
            self._validate_parameter("r_r[0]", self.r_r[0], self.f_bounds["r_r_0"].min, self.f_bounds["r_r_0"].max)
            self._validate_parameter("r_r[1]", self.r_r[1], self.f_bounds["r_r_1"].min, self.f_bounds["r_r_1"].max)
            self._validate_parameter(("r_l[0]", "r_r[0]"), (self.r_l[0], self.r_r[0]), self.f_bounds["r_l_0"].min, self.f_bounds["r_l_0"].max)
            self._validate_parameter(("r_l[1]", "r_r[1]"), (self.r_l[1], self.r_r[1]), self.f_bounds["r_l_0"].min, self.f_bounds["r_l_0"].max)
            self._validate_parameter("r_l", self.r_l, self.f_bounds["r_l_0"].min, self.f_bounds["r_l_0"].max)

            # Checks for manufacturing constraints
            self._validate_parameter("beta_l_1", self.beta_l_1, self.f_bounds["beta_l_1"].min, self.f_bounds["beta_l_1"].max)
            self._validate_parameter("beta_l_2", self.beta_l_2, self.f_bounds["beta_l_2"].min, self.f_bounds["beta_l_2"].max)
            self._validate_parameter("beta_r_1", self.beta_r_1, self.f_bounds["beta_r_1"].min, self.f_bounds["beta_r_1"].max)
            self._validate_parameter("beta_r_2", self.beta_r_2, self.f_bounds["beta_r_2"].min, self.f_bounds["beta_r_2"].max)

        except FeasibilityError as e:
            print(f"FeasibilityError: {e}")
            return False

        return True

    def create_geometry(self, segmentation_labels: bool = True, export: bool = True, gui: bool = False):
        """
        Create the UChannel1 geometry in gmsh using the parameters given in the constructor of UChannel1 and the
        calculated vertices.

        :param segmentation_labels: Extract the segmentation labels from the geometry.
        :param export: Export the geometry to the path given in the constructor of UChannel1
        :param gui: Launch the gmsh GUI

        :return: None
        """
        # Check fillet method
        self._check_fillet_method()

        gmsh.initialize()
        self.model = gmsh.model

        # 1. Create vertices
        self._create_vertices()

        # 2. Create edges
        self._create_edges()

        # 3. Create surfaces
        self._create_surfaces()

        # 4. Create fillets
        if self.fillet_method == "gmsh":
            if self.fl:
                fillet_edges_tags = [1, 2, 5, 6]
                self._create_fillet(
                    edges=fillet_edges_tags,
                    radii=[self.r_l[0], self.r_r[0], self.r_l[1], self.r_r[1], self.r_l[0], self.r_r[0], self.r_l[1],
                           self.r_r[1]]
                )
            else:
                fillet_edges_tags = [1, 2]
                self._create_fillet(
                    edges=fillet_edges_tags,
                    radii=[self.r_l[0], self.r_r[0], self.r_l[1], self.r_r[1]]
                )

        # Final gmsh model sync
        self.model.occ.synchronize()

        # Get the segmentation labels (auto-detects fillet + flange settings)
        if segmentation_labels:
            self.graph = self.face_adjacency.create_graph()
            self.get_segmentation_labels()

        # Export the geometry
        if export:
            self._export_geometry()
            if segmentation_labels:
                self._export_segmentation_labels()

        # Launch the GUI
        if gui:
            self._visualize_geometry()

        gmsh.finalize()

    def _create_vertices(self) -> None:
        """ Create all vertices of the geometry. """
        # Make sure we have a graph to work with
        if not hasattr(self, "graph"):
            self.graph = nx.Graph()

        # Collect all vertex attribute names
        vertex_names = [f"v{i}" for i in range(1, 13)]
        if self.fl:
            vertex_names.extend([f"v{i}" for i in range(13, 19)])

        # Flange radii tags. Which are needed for explicit filleting
        self.flange_radii_tags = [7, 8, 9, 10, 11, 12]

        for idx, name in enumerate(vertex_names, start=1):
            v = getattr(self, name)  # e.g. self.v1, self.v2 ...
            tag = self.model.occ.addPoint(v.x, v.y, v.z, tag=idx)   # Add a point in gmsh

            # Add a node to the graph with gmsh tag + coords
            self.graph.add_node(name, tag=tag, coords=(v.x, v.y, v.z))

    def _create_edges(self) -> None:
        """
        Create all edges of the geometry.

        Optionally instead of self.model.occ.addSpline one can use self.model.occ.addBSpline to create a B-spline curve.
        Pros:
            - The mid-point of the curve is not fixed in the x direction
        Cons:
            - The mid-point would not be on the curve. Rather a control point.
        """
        if not hasattr(self, "graph"):
            raise RuntimeError("Graph not initialized. Call _create_vertices first.")

        # Base edges
        edge_specs = [
            ("e1", "v1", "v5", "v2"),
            ("e2", "v3", "v6", "v4"),
            ("e3", "v1", "v3"),
            ("e4", "v2", "v4"),
            ("e5", "v7", "v11", "v8"),
            ("e6", "v9", "v12", "v10"),
            ("e7", "v1", "v7"),
            ("e8", "v2", "v8"),
            ("e9", "v3", "v9"),
            ("e10", "v4", "v10")
        ]

        # Flange edges
        if self.fl:
            edge_specs += [
                ("e11", "v13", "v17", "v14"),
                ("e12", "v7", "v13"),
                ("e13", "v8", "v14"),
                ("e14", "v9", "v15"),
                ("e15", "v10", "v16"),
                ("e16", "v15", "v18", "v16")
            ]

        for idx, spec in enumerate(edge_specs, start=1):
            edge_name, *nodes = spec  # Unpack: First is edge name, rest are vertex ids

            # Extract point tags from the graph
            point_tags = [self.graph.nodes[n]["tag"] for n in nodes]

            if len(point_tags) == 2:
                # Straight line between 2 points
                tag = self.model.occ.addLine(point_tags[0], point_tags[1], tag=idx)
                kind = "line"

            elif len(point_tags) == 3:
                # Spline through 3 points
                tag = self.model.occ.addSpline(point_tags, tag=idx)
                self.model.occ.remove([(0, point_tags[1])])
                kind = "spline"

            elif len(point_tags) == 4:
                # B-spline through 4 control points
                tag = self.model.occ.addBSpline(point_tags, tag=idx)
                kind = "bspline"

            else:
                raise ValueError(f"Unsupported number of vertices ({len(point_tags)}) in edge spec {spec}")

            # Add edge to graph
            self.graph.add_edge(nodes[0], nodes[-1], name=edge_name, tag=tag, kind=kind)

        # For the explicit fillet method, mark the helper edges as those will be specifically removed later
        if self.fillet_method == "explicit":
            if self.fl:
                helper_tags = (1, 2, 5, 6)
            else:
                helper_tags = (1, 2)
            # Set is_helper flag for helper edges in th graph
            for u, v, d in self.graph.edges(data=True):
                if d.get("tag") in helper_tags:
                    d["is_helper"] = True

    def _create_surfaces_procedural(self) -> None:
        """ Create all surfaces of the geometry using procedural construction """
        # Bottom Plane
        self.model.occ.addCurveLoop([1, 3, 2, 4], 10)
        self.model.occ.addBSplineFilling(10, 1)

        # Slant 2
        self.model.occ.addCurveLoop([7, 1, 8, 5], 20)
        self.model.occ.addBSplineFilling(20, 2)

        # Slant 1
        self.model.occ.addCurveLoop([9, 2, 10, 6], 30)
        self.model.occ.addBSplineFilling(30, 3)

        if self.fl:
            # Flange 2
            self.model.occ.addCurveLoop([12, 5, 13, 11], 40)
            self.model.occ.addBSplineFilling(40, 4)

            # Flange 1
            self.model.occ.addCurveLoop([14, 16, 15, 6], 50)
            self.model.occ.addBSplineFilling(50, 5)

    def _create_surfaces_explicit(self) -> None:
        """
        Create all surfaces of the geometry using explicit construction which includes explicit fillet construction.
        """
        self.model.occ.synchronize()

        # Calculate the bounding box of the geometry
        self._get_bounding_box()

        # UChannel 1 has the speciality that the fillet radius can be different for left and right as well as side 1
        # and side 2. Therefor we need to create a list of radii for each corner.
        radii = [self.r_l[0], self.r_l[1], self.r_r[0], self.r_r[1]]
        if self.fl:
            radii += [self.r_l[0], self.r_l[1], self.r_r[0], self.r_r[1]]

        # Create circular arcs at corners of the geometry using the graph representation of the geometry.
        self._graph_corner(self.flange_radii_tags, tolerance =abs((self.x_1 - self.x_2)) * 1.5, radii=radii)

        for vert_side in ["bottom", "top"]:
            # Connect the newly created circular arcs
            chains = self._connect_chain(vert_side)
            for chain in chains:
                if len(chain) !=5:
                    raise ValueError(f"Something went wrong with the chain. There should be 5 elements in the chain."
                                     f"Currently there are {len(chain)} elements in the chain.")
                offset = self.o_mp_2 if chain[-2] == "below" else self.o_mp_1
                # UChannel base only connects via lines. For UChannel 1 there is the speciality that we have spline
                # connections which can be offset using midpoints. We therefore need to replace the line with a spline.
                self._replace_line_with_spline(chain[-3], offset)

        self._remove_helper_edges()
        self._fragment_at_graph_nodes()
        self._rebuild_graph_from_occ()
        self._clean_graph()

        surfaces = self._build_surfaces_from_ordered_cycles()
        self._build_shell_from_surfaces(surfaces)

    def _replace_line_with_spline(self, line_tag: int, offset: tuple[float, float, float]) -> int:
        """
        Replace a straight line edge (identified by its tag) with a spline
        that passes through an offset midpoint.

        :param line_tag: OCC tag of the original line
        :param offset: (dx, dy, dz) to shift the midpoint
        :return: OCC tag of the new spline
        """
        # Find the edge in the graph that has this tag
        for u, v, data in self.graph.edges(data=True):
            if data.get("tag") == line_tag:
                start, end = u, v
                break
        else:
            raise ValueError(f"No edge found in graph with tag {line_tag}")

        coords1 = self.graph.nodes[start]["coords"]
        coords2 = self.graph.nodes[end]["coords"]

        # Midpoint offset
        mid = tuple((c1 + c2) / 2 + off for c1, c2, off in zip(coords1, coords2, offset))
        mid_tag = self.model.occ.addPoint(*mid)

        # Remove old line
        self.model.occ.remove([(1, line_tag)], recursive=False)

        # Create spline through [start, mid, end]
        spline_tag = self.model.occ.addSpline([
            self.graph.nodes[start]["tag"],
            mid_tag,
            self.graph.nodes[end]["tag"]
        ])

        self.model.occ.remove([(0, mid_tag)])

        # Update edge in graph
        self.graph.edges[start, end]["tag"] = spline_tag
        self.graph.edges[start, end]["kind"] = "chain_spline"

        return spline_tag


if __name__ == '__main__':
    # Define directories and paths
    work_dir = r"path/to/your/working/directory"
    geom_path = Path(work_dir) / 'uchannel1.step'
    label_path = Path(work_dir) / 'uchannel1.json'

    # Instantiate the geometry
    u1 = UChannel1(geom_path, label_path, x_1=100, x_2=105, y_l=50, y_r=50, z_l_1=33.3, z_l_2=33.3, z_r_1=33.3,
                   z_r_2=33.3,
                   o_mp_1=(1, 1, 1), o_mp_2=(1, 1, 1), r_l=(5, 5), r_r=(5, 5),
                   beta_l_1=15, beta_l_2=15, beta_r_1=15, beta_r_2=15, y_fl=50, fl=True)

    # Check the feasibility
    if u1.check_feasibility():
        # Create the geometry
        u1.fillet_method = "explicit"
        u1.create_geometry(segmentation_labels=True, export=True, gui=True)
        u1.visualize_segmentation_graph()