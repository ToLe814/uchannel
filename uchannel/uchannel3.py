from pathlib import Path
import math
import gmsh
import networkx as nx

from uchannel.uchannel_base import UChannelBase
from uchannel.error_handling import FeasibilityError


class UChannel3(UChannelBase):
    def __init__(self,
                 path: Path, label_path: Path, x_b1: float = 100, x_b2: float = 60, x_ad: float = 40, l_c1: float = 40, l_c2: float = 40,
                 x_t: float = 60, y_m: float = 50, alpha: float = 20, z_ad: float = 10, z_b: float = 20,
                 z_t: float = 20, beta_b: float = 15, beta_c: float = 15, beta_t: float = 15, y_fl: float = 10,
                 r: float = 5, ad: bool = True, fl: bool = False):
        """
        Class for generating the UChannel3 geometry in gmsh. A short description of the parameters is given below.

        The key parameter for feasibility control is x_b1. It can take any value. The authors recommend values between
        164 and 661 mm. The remaining parameters are limited by x_b1.

        :param path: Path to the step file where the geometry will be exported
        :param label_path: Path to the .json file where the segmentation labels will be exported
        :param x_b1: Length of the base 1 plane in x direction
        :param x_b2: Length of the base 2 plane in x direction
        :param x_ad: Length of the addendum in x direction
        :param l_c1: Length of the c1 plane in normal direction
        :param l_c2: Length of the c2 plane in normal direction
        :param x_t: Length of the top plane in x direction
        :param y_m: Width of the base plane in y direction
        :param alpha: Angle of the c1 plane in degrees
        :param z_ad: Height of the addendum in z direction from the base plane
        :param z_b: Height of the base plane in z direction
        :param z_t: Height of the top plane in z direction
        :param beta_b: Angle of the slant in section b and ad in degrees
        :param beta_c: Angle of the slant in section c in degrees
        :param beta_t: Angle of the slant in section t in degrees
        :param y_fl: Width of the flange in y direction
        :param r: Radius of the fillets
        :param ad: Boolean to decide if the addendum should be added
        :param fl: Boolean to decide if the flange should be added
        """
        super().__init__(path, label_path)

        # Parameters
        self.x_b1 = x_b1
        self.x_b2 = x_b2
        self.x_ad = x_ad
        self.l_c1 = l_c1
        self.l_c2 = l_c2
        self.x_t = x_t
        self.y_m = y_m
        self.y_fl = y_fl
        self.z_ad = z_ad
        self.z_b = z_b
        self.z_t = z_t
        self.alpha = math.radians(alpha)
        self.beta_b = math.radians(beta_b)
        self.beta_c = math.radians(beta_c)
        self.beta_t = math.radians(beta_t)
        self.r = r
        self.ad = ad
        self.fl = fl

        self.calculate_vertices()
        self.f_bounds = self._feasibility_bounds()  # Feasibility bounds

    def calculate_vertices(self):
        """
        Calculate the vertices of the UChannel3 geometry based on the parameters given in the constructor of UChannel3.
        The vertices order can be seen in the provided sketch.
        """
        # Vertices that construct the b2 plane
        self.v15 = self.Vertex(0., 0., 0.)
        self.v16 = self.Vertex(0., -self.y_m, 0.)

        # Vertices that construct the b1 plane
        self.v9 = self.Vertex(-self.x_b2, 0., 0.)
        self.v10 = self.Vertex(-self.x_b2, -self.y_m, 0.)
        self.v3 = self.Vertex(-self.x_b2 - self.x_b1, 0., 0.)
        self.v4 = self.Vertex(-self.x_b2 - self.x_b1, -self.y_m, 0.)

        # Vertices that construct the c1 plane
        self.v21 = self.Vertex(self.l_c1 * math.cos(self.alpha), 0., self.l_c1 * math.sin(self.alpha))
        self.v22 = self.Vertex(self.l_c1 * math.cos(self.alpha), -self.y_m, self.l_c1 * math.sin(self.alpha))

        # Vertices that construct the top plane
        x_c = (self.l_c1 + self.l_c2) * math.cos(self.alpha)
        z_c = (self.l_c1 + self.l_c2) * math.sin(self.alpha)

        self.v26 = self.Vertex(x_c, 0., z_c)
        self.v27 = self.Vertex(x_c, -self.y_m, z_c)
        self.v30 = self.Vertex(x_c + self.x_t, 0., z_c)
        self.v31 = self.Vertex(x_c + self.x_t, -self.y_m, z_c)

        # Vertices that construct the s1 slant
        y_s1_b = self.z_b * math.tan(self.beta_b)
        y_s1_c = self.z_b * math.tan(self.beta_c)
        y_s1_t = self.z_b * math.tan(self.beta_t)
        self.v8 = self.Vertex(self.v9.x, self.v9.y + y_s1_b, self.v9.z + self.z_b)
        self.v2 = self.Vertex(self.v8.x - self.x_b1, self.v8.y, self.v8.z)
        self.v38 = self.Vertex(self.v2.x - self.x_ad, self.v2.y, self.v2.z)
        self.v29 = self.Vertex(self.v30.x, self.v30.y + y_s1_t, self.v30.z + self.z_t)
        self.v25 = self.Vertex(self.v29.x - self.x_t, self.v29.y, self.v29.z)
        self.v20 = self.Vertex(self.v25.x - self.l_c2 * math.cos(self.alpha), self.v30.y + y_s1_c, self.v25.z)
        self.v14 = self.Vertex(self.v20.x - self.l_c1 * math.cos(self.alpha), self.v20.y,
                               self.z_b + ((self.v15.z - self.v9.x) / (self.v20.x - self.v9.x)) * (self.v20.z - self.v8.z))

        # Vertices that construct the s1 slant and addendum
        self.v34 = self.Vertex(self.v29.x, self.v29.y + self.y_fl, self.v29.z)
        self.v33 = self.Vertex(self.v25.x, self.v25.y + self.y_fl, self.v25.z)
        self.v19 = self.Vertex(self.v20.x, self.v20.y + self.y_fl, self.v20.z)
        self.v13 = self.Vertex(self.v14.x, self.v14.y + self.y_fl, self.v14.z)
        self.v7 = self.Vertex(self.v8.x, self.v8.y + self.y_fl, self.v8.z)
        self.v1 = self.Vertex(self.v2.x, self.v2.y + self.y_fl, self.v2.z)
        self.v37 = self.Vertex(self.v38.x, self.v38.y + self.y_fl, self.v38.z)

        # Vertices that construct the s2 slant and addendum
        self.v11 = self.Vertex(self.v10.x, self.v10.y - y_s1_b, self.v10.z + self.z_b)
        self.v5 = self.Vertex(self.v11.x - self.x_b1, self.v11.y, self.v11.z)
        self.v41 = self.Vertex(self.v5.x - self.x_ad, self.v5.y, self.v5.z)
        self.v32 = self.Vertex(self.v31.x, self.v31.y - y_s1_t, self.v31.z + self.z_t)
        self.v28 = self.Vertex(self.v32.x - self.x_t, self.v32.y, self.v32.z)
        self.v23 = self.Vertex(self.v28.x - self.l_c2 * math.cos(self.alpha), self.v31.y - y_s1_c, self.v28.z)
        self.v17 = self.Vertex(self.v23.x - self.l_c1 * math.cos(self.alpha), self.v23.y,
                               self.z_b + ((self.v15.z - self.v9.x) / (self.v20.x - self.v9.x)) * (self.v20.z - self.v8.z))
        self.v36 = self.Vertex(self.v32.x, self.v32.y - self.y_fl, self.v32.z)
        self.v35 = self.Vertex(self.v28.x, self.v28.y - self.y_fl, self.v28.z)
        self.v24 = self.Vertex(self.v23.x, self.v23.y - self.y_fl, self.v23.z)
        self.v18 = self.Vertex(self.v17.x, self.v17.y - self.y_fl, self.v17.z)
        self.v12 = self.Vertex(self.v11.x, self.v11.y - self.y_fl, self.v11.z)
        self.v6 = self.Vertex(self.v5.x, self.v5.y - self.y_fl, self.v5.z)
        self.v42 = self.Vertex(self.v41.x, self.v41.y - self.y_fl, self.v41.z)

        # Vertices that construct the ad plane
        self.v39 = self.Vertex(-self.x_b2 - self.x_b1 - self.x_ad, 0., self.z_ad)
        self.v40 = self.Vertex(-self.x_b2 - self.x_b1 - self.x_ad, -self.y_m, self.z_ad)

        # Helper points for spline in ad
        self.v43 = self.Vertex(self.v3.x - 0.25 * self.x_ad, self.v3.y, self.v3.z + 0.1 * self.z_ad)
        self.v44 = self.Vertex(self.v3.x - 0.50 * self.x_ad, self.v3.y, self.v3.z + 0.5 * self.z_ad)
        self.v45 = self.Vertex(self.v3.x - 0.75 * self.x_ad, self.v3.y, self.v3.z + 0.9 * self.z_ad)
        self.v46 = self.Vertex(self.v43.x, self.v43.y - self.y_m, self.v43.z)
        self.v47 = self.Vertex(self.v44.x, self.v44.y - self.y_m, self.v44.z)
        self.v48 = self.Vertex(self.v45.x, self.v45.y - self.y_m, self.v45.z)

    def _feasibility_bounds(self) -> dict:
        """
        Bounds for the feasibility check. Can be used as an input for an LHS sampler.
        """
        bounds = {
            "x_b2": self.Bounds(self.x_b1 * 1 / 10, self.x_b1 * 3 / 5),
            "x_ad": self.Bounds(self.x_b1 * 1 / 5, self.x_b1 * 2 / 5),
            "l_c1": self.Bounds(self.x_b1 * 1 / 7, self.x_b1 * 1 / 2),
            "l_c2": self.Bounds(self.x_b1 * 1 / 7, self.x_b1 * 3 / 5),
            "x_t": self.Bounds(self.x_b1 * 1 / 10, self.x_b1 * 3 / 5),
            "y_m": self.Bounds(self.x_b1 * 1 / 5, self.x_b1 * 1 / 2),
            "alpha": self.Bounds(0, math.radians(20)),
            "z_ad": self.Bounds(0, self.x_b1 * 1 / 20),
            "z_b": self.Bounds(self.x_b1 * 1 / 10, self.x_b1 * 1 / 5),
            "z_t": self.Bounds(self.x_b1 * 1 / 15, self.x_b1 * 1 / 5),
            "beta_b": self.Bounds(math.radians(5), math.radians(15)),
            "beta_c": self.Bounds(math.radians(5), math.radians(15)),
            "beta_t": self.Bounds(math.radians(5), math.radians(15)),
            "y_fl": self.Bounds(self.x_b1 * 1 / 20, self.x_b1 * 1 / 10),
            "r": self.Bounds(self.x_b1 * 1 / 100, self.x_b1 * 1 / 30),
            "ad": self.Bounds(True, False),
            "fl": self.Bounds(True, False)
        }
        return bounds

    def check_feasibility(self) -> bool:
        """
        Check if the parameters are feasible for geometry generation.
        The idea is that x_b1 can be any value, but the other parameters are limited by the value of x_b1.

        :return: True if the parameters are feasible, False otherwise
        """
        try:
            # Validate individual parameters against their constraints
            # Checks for realistic proportions
            self._validate_parameter('x_b2', self.x_b2, self.f_bounds["x_b2"].min, self.f_bounds["x_b2"].max)
            self._validate_parameter('x_ad', self.x_ad, self.f_bounds["x_ad"].min, self.f_bounds["x_ad"].max)
            self._validate_parameter('l_c1', self.l_c1, self.f_bounds["l_c1"].min, self.f_bounds["l_c1"].max)
            self._validate_parameter('l_c2', self.l_c2, self.f_bounds["l_c2"].min, self.f_bounds["l_c2"].max)
            self._validate_parameter('x_t', self.x_t, self.f_bounds["x_t"].min, self.f_bounds["x_t"].max)
            self._validate_parameter('y_m', self.y_m, self.f_bounds["y_m"].min, self.f_bounds["y_m"].max)
            self._validate_parameter('y_fl', self.y_fl, self.f_bounds["y_fl"].min, self.f_bounds["y_fl"].max)
            self._validate_parameter('z_b', self.z_b, self.f_bounds["z_b"].min, self.f_bounds["z_b"].max)
            self._validate_parameter('z_t', self.z_t, self.f_bounds["z_t"].min, self.f_bounds["z_t"].max)

            # Checks for geometry generation constraints (Overlaps, Intersections, Gmsh Feasibility etc.)
            self._validate_parameter('r', self.r, self.f_bounds["r"].min, self.f_bounds["r"].max)
            self._validate_parameter('z_ad', self.z_ad, self.f_bounds["z_ad"].min, self.f_bounds["z_ad"].max)

            # Checks for manufacturing constraints
            self._validate_parameter('alpha', self.alpha, self.f_bounds["alpha"].min, self.f_bounds["alpha"].max)
            self._validate_parameter('beta_b', self.beta_b, self.f_bounds["beta_b"].min, self.f_bounds["beta_b"].max)
            self._validate_parameter('beta_c', self.beta_c, self.f_bounds["beta_c"].min, self.f_bounds["beta_c"].max)
            self._validate_parameter('beta_t', self.beta_t, self.f_bounds["beta_t"].min, self.f_bounds["beta_t"].max)

        except FeasibilityError as e:
            print(f"FeasibilityError: {e}")
            return False

        return True

    def create_geometry(self, segmentation_labels: bool = True, export=True, gui=False):
        """
        Create the UChannel3 geometry in gmsh using the parameters given in the constructor of UChannel3 and the
        calculated vertices.

        :param segmentation_labels: Extract the segmentation labels from the geometry.
        :param export: Export the geometry to the path given in the constructor of UChannel3
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
            # Edges that need fillets weather fl, ad or not
            fillet_edges_tags = [6, 7, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41,
                                 42, 43, 44, 45, 46, 47]

            if self.fl:
                fillet_edges_tags.extend([10, 6, 16, 20, 33, 34, 48, 49])
            if self.ad:
                fillet_edges_tags.extend([62, 63, 64, 65])

            # Conduct gmsh filleting
            self._create_fillet(edges=fillet_edges_tags, radii=[self.r])

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

    def _create_vertices(self):
        """ Create all vertices of the geometry. """
        # Make sure we have a graph to work with
        if not hasattr(self, "graph"):
            self.graph = nx.Graph()

        # Base vertices (always added)
        base_points = list(range(1, 33))

        if self.fl:
            base_points.extend(list(range(33, 37)))
        if self.ad:
            base_points.extend([37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])

        # Flange radii tags. Which are needed for explicit filleting
        if self.fl and self.ad:
            self.flange_radii_tags = [38, 2, 8, 14, 20, 25, 29, 41, 5, 11, 17, 23, 28, 32]
        if self.fl and not self.ad:
            self.flange_radii_tags = [2, 8, 14, 20, 25, 29, 5, 11, 17, 23, 28, 32]
        if not self.fl and not self.ad:
            self.flange_radii_tags = [2, 8, 14, 20, 5, 11, 17, 23]
        if not self.fl and self.ad:
            self.flange_radii_tags = [38, 2, 8, 14, 41, 5, 11, 17, 20, 23]

        for tag in base_points:
            v = getattr(self, f"v{tag}")  # e.g. self.v1, self.v2...
            tag = self.model.occ.addPoint(*v, tag=tag)  # Add point in gmsh

            # Add a node to the graph with gmsh tag + coords
            self.graph.add_node(f"v{tag}", tag=tag, coords=tuple(v))

    def _create_edges(self):
        """ Create all edges of the geometry. """
        if not hasattr(self, "graph"):
            raise RuntimeError("Graph not initialized. Call _create_vertices first.")

        # Base edges
        edge_specs = [
            ("e1", 1, 2, 1),
            ("e2", 2, 3, 2),
            ("e3", 3, 4, 3),
            ("e4", 4, 5, 4),
            ("e5", 5, 6, 5),
            ("e6", 7, 8, 6),
            ("e7", 8, 9, 7),
            ("e8", 9, 10, 8),
            ("e9", 10, 11, 9),
            ("e10", 11, 12, 10),
            ("e11", 13, 14, 11),
            ("e12", 14, 15, 12),
            ("e13", 15, 16, 13),
            ("e14", 16, 17, 14),
            ("e15", 17, 18, 15),
            ("e16", 19, 20, 16),
            ("e17", 20, 21, 17),
            ("e18", 21, 22, 18),
            ("e19", 22, 23, 19),
            ("e20", 23, 24, 20),
            ("e21", 25, 26, 21),
            ("e22", 26, 27, 22),
            ("e23", 27, 28, 23),
            ("e24", 29, 30, 24),
            ("e25", 30, 31, 25),
            ("e26", 31, 32, 26),
            ("e27", 1, 7, 27),
            ("e28", 7, 13, 28),
            ("e29", 13, 19, 29),
            ("e30", 2, 8, 30),
            ("e31", 8, 14, 31),
            ("e32", 14, 20, 32),
            ("e33", 20, 25, 33),
            ("e34", 25, 29, 34),
            ("e35", 3, 9, 35),
            ("e36", 9, 15, 36),
            ("e37", 15, 21, 37),
            ("e38", 21, 26, 38),
            ("e39", 26, 30, 39),
            ("e40", 4, 10, 40),
            ("e41", 10, 16, 41),
            ("e42", 16, 22, 42),
            ("e43", 22, 27, 43),
            ("e44", 27, 31, 44),
            ("e45", 5, 11, 45),
            ("e46", 11, 17, 46),
            ("e47", 17, 23, 47),
            ("e48", 23, 28, 48),
            ("e49", 28, 32, 49),
            ("e50", 6, 12, 50),
            ("e51", 12, 18, 51),
            ("e52", 18, 24, 52),
        ]

        # Flange edges
        if self.fl:
            edge_specs += [
                ("e53", 19, 33, 53),
                ("e54", 33, 34, 54),
                ("e55", 33, 25, 55),
                ("e56", 34, 29, 56),
                ("e57", 24, 35, 57),
                ("e58", 35, 36, 58),
                ("e59", 35, 28, 59),
                ("e60", 36, 32, 60),
            ]

        # Addendum edges
        if self.ad:
            edge_specs += [
                ("e61", 1, 37, 61),
                ("e62", 2, 38, 62),
                # Splines instead of lines
                ("s63", [3, 43, 44, 45, 39], 63),
                ("s64", [4, 46, 47, 48, 40], 64),
                ("e65", 5, 41, 65),
                ("e66", 6, 42, 66),
                ("e67", 37, 38, 67),
                ("e68", 38, 39, 68),
                ("e69", 39, 40, 69),
                ("e70", 40, 41, 70),
                ("e71", 41, 42, 71),
            ]

            # Create edges in gmsh and add them to the graph
        for edge in edge_specs:
            if len(edge) == 4:
                edge_name, n1_tag, n2_tag, etag = edge
                tag = self.model.occ.addLine(n1_tag, n2_tag, tag=etag)
                self.graph.add_edge(f"v{n1_tag}", f"v{n2_tag}", name=edge_name, tag=tag, kind="line")
            elif len(edge) == 3:
                edge_name, points, etag = edge
                tag = self.model.occ.addSpline(points, tag=etag)
                self.graph.add_edge(f"v{points[0]}", f"v{points[-1]}", name=edge_name, tag=tag, kind="spline")

        # For the explicit fillet method, mark the helper edges as those will be specifically removed later
        if self.fillet_method == "explicit":
            if self.fl and self.ad:
                self.helper_tags = (30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 62, 63, 64, 65)
            if self.fl and not self.ad:
                self.helper_tags = (30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49)
            if not self.fl and not self.ad:
                self.helper_tags = (30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47)
            if not self.fl and self.ad:
                self.helper_tags = (30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 62, 63, 64, 65)
            # Set is_helper flag for helper edges in th graph
            for u, v, d in self.graph.edges(data=True):
                if d.get("tag") in self.helper_tags:
                    d["is_helper"] = True

    def _create_surfaces_procedural(self):
        """ Create all surfaces of the geometry using procedural construction """
        self.model.occ.addCurveLoop([1, 30, 6, 27], 1)
        self.model.occ.addSurfaceFilling(1, 1, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([2, 35, 7, 30], 3)
        self.model.occ.addSurfaceFilling(3, 2, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([3, 40, 8, 35], 5)
        self.model.occ.addSurfaceFilling(5, 3, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([4, 45, 9, 40], 7)
        self.model.occ.addSurfaceFilling(7, 4, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([5, 50, 10, 45], 9)
        self.model.occ.addSurfaceFilling(9, 5, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([6, 31, 11, 28], 11)
        self.model.occ.addSurfaceFilling(11, 6, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([7, 36, 12, 31], 13)
        self.model.occ.addSurfaceFilling(13, 7, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([8, 41, 13, 36], 15)
        self.model.occ.addSurfaceFilling(15, 8, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([9, 46, 14, 41], 17)
        self.model.occ.addSurfaceFilling(17, 9, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([10, 51, 15, 46], 19)
        self.model.occ.addSurfaceFilling(19, 10, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([11, 32, 16, 29], 21)
        self.model.occ.addSurfaceFilling(21, 11, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([12, 37, 17, 32], 23)
        self.model.occ.addSurfaceFilling(23, 12, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([13, 42, 18, 37], 25)
        self.model.occ.addSurfaceFilling(25, 13, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([14, 47, 19, 42], 27)
        self.model.occ.addSurfaceFilling(27, 14, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([15, 52, 20, 47], 29)
        self.model.occ.addSurfaceFilling(29, 15, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([17, 38, 21, 33], 31)
        self.model.occ.addSurfaceFilling(31, 16, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([18, 43, 22, 38], 33)
        self.model.occ.addSurfaceFilling(33, 17, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([19, 48, 23, 43], 35)
        self.model.occ.addSurfaceFilling(35, 18, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([21, 39, 24, 34], 37)
        self.model.occ.addSurfaceFilling(37, 19, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([22, 44, 25, 39], 39)
        self.model.occ.addSurfaceFilling(39, 20, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([23, 49, 26, 44], 41)
        self.model.occ.addSurfaceFilling(41, 21, degree=2, maxDegree=3)

        if self.fl:
            self.model.occ.addCurveLoop([16, 33, 55, 53], 43)
            self.model.occ.addSurfaceFilling(43, 22, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([55, 34, 56, 54], 45)
            self.model.occ.addSurfaceFilling(45, 23, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([20, 57, 59, 48], 47)
            self.model.occ.addSurfaceFilling(47, 24, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([59, 58, 60, 49], 49)
            self.model.occ.addSurfaceFilling(49, 25, degree=2, maxDegree=3)

        if self.ad:
            self.model.occ.addCurveLoop([67, 62, 1, 61], 51)
            self.model.occ.addSurfaceFilling(51, 26, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([68, 63, 2, 62], 53)
            self.model.occ.addSurfaceFilling(53, 27, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([69, 64, 3, 63], 55)
            self.model.occ.addSurfaceFilling(55, 28, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([70, 65, 4, 64], 57)
            self.model.occ.addSurfaceFilling(57, 29, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([71, 66, 5, 65], 59)
            self.model.occ.addSurfaceFilling(59, 30, degree=2, maxDegree=3)

    def _create_surfaces_explicit(self) -> None:
        """
        Create all surfaces of the geometry using explicit construction which includes explicit fillet construction.
        """
        self.model.occ.synchronize()

        # Calculate the bounding box of the geometry
        self._get_bounding_box()

        # Create circular arcs at corners of the geometry using the graph representation of the geometry.
        self._graph_corner(self.flange_radii_tags)


        # Connect the newly created circular arcs
        self._connect_fillet_nodes_on_yz_planes(tolerance=10)
        # Fillet the newly created circular arc junctions
        # Connect fillet edges with lines
        self._fillet_support_junctions()
        self._connect_all_junctions_with_bspline()
        self._connect_chain("top")
        self._connect_chain("bottom")

        if not self.fl:
            _, pt_tags = self.model.getAdjacencies(1, 33)
            a_tag = max(pt_tags, key=lambda pt: self.model.getValue(0, pt, [])[0])

            _, pt_tags = self.model.getAdjacencies(1, 48)
            b_tag = max(pt_tags, key=lambda pt: self.model.getValue(0, pt, [])[0])

            self.model.occ.remove([(1, 33), (1, 48)])
            self.graph.edges["v20_fillet_yz_top_is_flange", "v20_fillet_yz_bottom_is_flange"]["is_helper"] = False
            self.graph.edges["v23_fillet_yz_bottom_is_flange", "v23_fillet_yz_top_is_flange"]["is_helper"] = False

            a_end = self.graph.nodes["v20_fillet_yz_bottom_is_flange"]["tag"]
            b_end = self.graph.nodes["v23_fillet_yz_bottom_is_flange"]["tag"]

            self.model.occ.addLine(a_end, a_tag, 33)
            self.model.occ.addLine(b_end, b_tag, 48)

        if self.ad:
            for u, v, d in self.graph.edges(data=True):
                if d["name"] == "arc_v3_yz" or d["name"] == "arc_v4_yz":
                    d["is_helper"] = False

        self._remove_helper_edges()
        self._fragment_at_graph_nodes()

        self._rebuild_graph_from_occ()

        self._clean_graph()
        surfaces = self._build_surfaces_from_ordered_cycles()
        self._build_shell_from_surfaces(surfaces)

if __name__ == '__main__':
    # Define directories and paths
    work_dir = r"path/to/your/working/directory"
    geom_path = Path(work_dir) / 'uchannel3.step'
    label_path = Path(work_dir) / 'uchannel3.json'

    # Instantiate the geometry
    u3 = UChannel3(geom_path, label_path, x_b1=100, x_b2=10, x_ad=20, l_c1=14.29, l_c2=14.92, x_t=10, y_m=20, alpha=20,
                   z_ad=0, z_b=10, z_t=6.7, beta_b=5, beta_c=15, beta_t=5, y_fl=5, r=3.0, ad=True, fl=True)

    # Check the feasibility
    if u3.check_feasibility():
        # Create the geometry
        u3.fillet_method = "explicit"
        u3.create_geometry(segmentation_labels=True, export=True, gui=False)
        u3.visualize_segmentation_graph()