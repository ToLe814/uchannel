from pathlib import Path
import math
import gmsh
import networkx as nx

from uchannel.uchannel_base import UChannelBase
from uchannel.error_handling import FeasibilityError


class UChannel2(UChannelBase):
    def __init__(self, path: Path, label_path: Path, x_m: float = 100, x_l: float = 25, x_r: float = 25, ll: float = 12.5,
                 lr: float = 12.5, z_m: float = 16.67, z_l: float = 11.12, z_r: float = 11.12, y_m: float = 20,
                 y_l: float = 10, y_r: float = 10, y_fl: float = 5, alpha_l: float = 0, alpha_r: float = 0,
                 beta_l: float = 5, beta_r: float = 5, beta_m: float = 5, r: float = 0.4, fl: bool = True,
                 la: bool = True, ra: bool = True):
        """
        Class for generating the UChannel2 geometry in gmsh. A short description of the parameters is given below.

        The key parameter for feasibility control is x_m. It can take any value. The authors recommend values between
        128 and 700 mm. The remaining parameters are limited by x_m.

        :param path: Path to the .step file where the geometry will be exported
        :param label_path: Path to the .json file where the segmentation labels will be exported
        :param x_m: Length of the middle plane in the x direction
        :param x_l: Length of the left addendum in the x direction
        :param x_r: Length of the right addendum in the x direction
        :param ll: Length of the left slanted plane along its tangent
        :param lr: Length of the right slanted plane along its tangent
        :param z_m: Height of the middle plane in the z direction
        :param z_l: Height of the left addendum in the z direction
        :param z_r: Height of the right addendum in the z direction
        :param y_m: Width of the middle plane in the y direction
        :param y_l: Width of the left addendum in the y direction
        :param y_r: Width of the right addendum in the y direction
        :param y_fl: Width of the flange in the y direction
        :param alpha_l: Angle of the left slanted plane
        :param alpha_r: Angle of the right slanted plane
        :param beta_l: Angle of the slant of the left plane in degrees
        :param beta_r: Angle of the slant of the right plane in degrees
        :param beta_m: Angle of the slant of the middle plane in degrees
        :param r: Radius of the fillets. Radius is the same for all fillets.
        :param fl: Boolean to determine if the slant addendum is present
        :param la: Boolean to determine if the left addendum is present
        :param ra: Boolean to determine if the right addendum is present
        """
        super().__init__(path, label_path)

        # Parameters
        self.x_m: float = x_m
        self.x_l = x_l
        self.x_r = x_r
        self.ll = ll
        self.lr = lr
        self.z_m = z_m
        self.z_l = z_l
        self.z_r = z_r
        self.y_m = y_m
        self.y_l = y_l
        self.y_r = y_r
        self.y_fl = y_fl
        self.alpha_l = math.radians(alpha_l)
        self.alpha_r = math.radians(alpha_r)
        self.beta_l = math.radians(beta_l)
        self.beta_r = math.radians(beta_r)
        self.beta_m = math.radians(beta_m)
        self.r = r
        self.fl = fl
        self.la = la
        self.ra = ra

        self._calculate_vertices()
        self.f_bounds = self._feasibility_bounds()  # Feasibility bounds

    def _calculate_vertices(self) -> None:
        """
        Calculate the vertices of the UChannel2 geometry based on the parameters given in the constructor of UChannel2.
        The vertices order can be seen in the provided sketch.
        """
        # Vertices that construct the middle plane m
        self.v1 = self.Vertex(0., 0., 0.)
        self.v2 = self.Vertex(self.x_m, 0., 0.)
        self.v3 = self.Vertex(self.x_m, -self.y_m, 0.)
        self.v4 = self.Vertex(0., -self.y_m, 0.)

        # Vertices that construct the left plane l
        x_c_l = math.cos(self.alpha_l)*self.ll
        z_c_l = math.sin(self.alpha_l)*self.ll
        y_c_l = (self.y_m - self.y_l)/2
        self.v5 = self.Vertex(-x_c_l, -y_c_l, z_c_l)
        self.v6 = self.Vertex(-x_c_l - self.x_l, -y_c_l, z_c_l)
        self.v7 = self.Vertex(-x_c_l - self.x_l, -y_c_l-self.y_l, z_c_l)
        self.v8 = self.Vertex(-x_c_l, -y_c_l-self.y_l, z_c_l)

        # Vertices that construct the right plane r
        x_c_r = math.cos(self.alpha_r) * self.lr
        z_c_r = math.sin(self.alpha_r) * self.lr
        y_c_r = (self.y_m - self.y_r) / 2
        self.v9 = self.Vertex(self.x_m + x_c_r, -y_c_r, z_c_r)
        self.v10 = self.Vertex(self.x_m + x_c_r + self.x_r, -y_c_r, z_c_r)
        self.v11 = self.Vertex(self.x_m + x_c_r + self.x_r, -y_c_r - self.y_r, z_c_r)
        self.v12 = self.Vertex(self.x_m + x_c_r, -y_c_r - self.y_r, z_c_r)

        # Vertices that construct the slant of the left plane l
        y_s_l = math.tan(self.beta_l) * self.z_l
        self.v13 = self.Vertex(self.v6.x, self.v6.y + y_s_l, self.v6.z + self.z_l)
        self.v14 = self.Vertex(self.v13.x, self.v13.y + self.y_fl, self.v13.z)
        self.v15 = self.Vertex(self.v13.x + self.x_l, self.v13.y, self.v13.z)
        self.v16 = self.Vertex(self.v15.x, self.v15.y + self.y_fl, self.v15.z)

        # Vertices that construct the slant of the middle plane m
        y_s_m = math.tan(self.beta_m) * self.z_m
        self.v17 = self.Vertex(self.v1.x, self.v1.y + y_s_m, self.v1.z + self.z_m)
        self.v18 = self.Vertex(self.v17.x, self.v17.y + self.y_fl, self.v17.z)
        self.v19 = self.Vertex(self.v17.x + self.x_m, self.v17.y, self.v17.z)
        self.v20 = self.Vertex(self.v19.x, self.v19.y + self.y_fl, self.v19.z)

        # Vertices that construct the slant of the right plane r
        y_s_r = math.tan(self.beta_r) * self.z_r
        self.v23 = self.Vertex(self.v10.x, self.v10.y + y_s_r, self.v10.z + self.z_r)
        self.v21 = self.Vertex(self.v23.x - self.x_r, self.v23.y, self.v23.z)
        self.v22 = self.Vertex(self.v21.x, self.v21.y + self.y_fl, self.v21.z)
        self.v24 = self.Vertex(self.v23.x, self.v23.y + self.y_fl, self.v23.z)

        # Vertices that construct the slant of the left plane l
        self.v25 = self.Vertex(self.v7.x, self.v7.y - y_s_l, self.v7.z + self.z_l)
        self.v26 = self.Vertex(self.v25.x, self.v25.y - self.y_fl, self.v25.z)
        self.v27 = self.Vertex(self.v25.x + self.x_l, self.v25.y, self.v25.z)
        self.v28 = self.Vertex(self.v27.x, self.v27.y - self.y_fl, self.v27.z)

        # Vertices that construct the slant of the middle plane m
        self.v29 = self.Vertex(self.v4.x, self.v4.y - y_s_m, self.v4.z + self.z_m)
        self.v30 = self.Vertex(self.v29.x, self.v29.y - self.y_fl, self.v29.z)
        self.v31 = self.Vertex(self.v29.x + self.x_m, self.v29.y, self.v29.z)
        self.v32 = self.Vertex(self.v31.x, self.v31.y - self.y_fl, self.v31.z)

        # Vertices that construct the slant of the right plane r
        self.v35 = self.Vertex(self.v11.x, self.v11.y - y_s_r, self.v11.z + self.z_r)
        self.v33 = self.Vertex(self.v35.x - self.x_r, self.v35.y, self.v35.z)
        self.v34 = self.Vertex(self.v33.x, self.v33.y - self.y_fl, self.v33.z)
        self.v36 = self.Vertex(self.v35.x, self.v35.y - self.y_fl, self.v35.z)

    def _feasibility_bounds(self) -> dict:
        """
        Bounds for the feasibility check. Can be used as an input for an LHS sampler.
        """
        bounds = {
            "x_l": self.Bounds(1/4*self.x_m, 3/4*self.x_m),
            "x_r": self.Bounds(1/4*self.x_m, 3/4*self.x_m),
            "ll": self.Bounds(3/8*self.x_m, 1/2*self.x_m),
            "lr": self.Bounds(3/8*self.x_m, 1/2*self.x_m),
            "z_m": self.Bounds(1/6*self.x_m, 1/2*self.x_m),
            "z_l": self.Bounds(1/9*self.x_m, 2/6*self.x_m),
            "z_r": self.Bounds(1/9*self.x_m, 2/6*self.x_m),
            "y_m": self.Bounds(1/5*self.x_m, 3/8*self.x_m),
            "y_l": self.Bounds(1/8*self.x_m, 9/16*self.x_m),
            "y_r": self.Bounds(1/8*self.x_m, 9/16*self.x_m),
            "y_fl": self.Bounds(1/20*self.x_m, 3/20*self.x_m),
            "alpha_l": self.Bounds(0, math.radians(30)),
            "alpha_r": self.Bounds(0, math.radians(30)),
            "beta_l": self.Bounds(math.radians(5), math.radians(15)),
            "beta_r": self.Bounds(math.radians(5), math.radians(15)),
            "beta_m": self.Bounds(math.radians(5), math.radians(15)),
            "r": self.Bounds(1/250*self.x_m, 1/25*self.x_m),
            "fl": self.Bounds(True, False),
            "la": self.Bounds(True, False),
            "ra": self.Bounds(True, False)
        }
        return bounds

    def check_feasibility(self) -> bool:
        """
        Check if the parameters are feasible for geometry generation.
        The idea is that x_m can be any value, but the other parameters are limited by the value of x_m.

        :return: True if the parameters are feasible, False otherwise
        """
        try:
            # Validate individual parameters against their constraints
            # Checks for realistic proportions
            self._validate_parameter("x_l", self.x_l, self.f_bounds["x_l"].min, self.f_bounds["x_l"].max)
            self._validate_parameter("x_r", self.x_r, self.f_bounds["x_r"].min, self.f_bounds["x_r"].max)
            self._validate_parameter("ll", self.ll, self.f_bounds["ll"].min, self.f_bounds["ll"].max)
            self._validate_parameter("lr", self.lr, self.f_bounds["lr"].min, self.f_bounds["lr"].max)
            self._validate_parameter("z_m", self.z_m, self.f_bounds["z_m"].min, self.f_bounds["z_m"].max)
            self._validate_parameter("z_l", self.z_l, self.f_bounds["z_l"].min, self.f_bounds["z_l"].max)
            self._validate_parameter("z_r", self.z_r, self.f_bounds["z_r"].min, self.f_bounds["z_r"].max)
            self._validate_parameter("y_m", self.y_m, self.f_bounds["y_m"].min, self.f_bounds["y_m"].max)
            self._validate_parameter("y_l", self.y_l, self.f_bounds["y_l"].min, self.f_bounds["y_l"].max)
            self._validate_parameter("y_r", self.y_r, self.f_bounds["y_r"].min, self.f_bounds["y_r"].max)
            if self.fl:
                self._validate_parameter("y_fl", self.y_fl, self.f_bounds["y_fl"].min, self.f_bounds["y_fl"].max)
            self._validate_parameter("alpha_l", self.alpha_l, self.f_bounds["alpha_l"].min, self.f_bounds["alpha_l"].max)
            self._validate_parameter("alpha_r", self.alpha_r, self.f_bounds["alpha_r"].min, self.f_bounds["alpha_r"].max)

            # Checks for geometry generation constraints (Overlaps, Intersections, Gmsh Feasibility etc.)
            self._validate_parameter("r", self.r, self.f_bounds["r"].min, self.f_bounds["r"].max)

            # Checks for manufacturing constraints
            self._validate_parameter("beta_l", self.beta_l, self.f_bounds["beta_l"].min, self.f_bounds["beta_l"].max)
            self._validate_parameter("beta_r", self.beta_r, self.f_bounds["beta_r"].min, self.f_bounds["beta_r"].max)
            self._validate_parameter("beta_m", self.beta_m, self.f_bounds["beta_m"].min, self.f_bounds["beta_m"].max)

        except FeasibilityError as e:
            print(f"FeasibilityError: {e}")
            return False

        return True

    def create_geometry(self, segmentation_labels: bool = True, export: bool = True, gui: bool = False) -> None:
        """
        Create the UChannel2 geometry in gmsh using the parameters given in the constructor of UChannel2 and the
        calculated vertices.

        :param segmentation_labels: Extract the segmentation labels from the geometry.
        :param export: Export the geometry to the path given in the constructor of UChannel2
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
            # Edges that need fillets weather fl, la, ra or not
            fillet_edges_tags = [1, 2, 3, 4, 5, 10, 21, 43, 11, 16, 23, 45]

            if self.la:
                fillet_edges_tags.extend([6, 8, 19, 41, 9])
            if self.ra:
                fillet_edges_tags.extend([12, 14, 15, 25, 47])

            if self.fl:
                if self.la:
                    fillet_edges_tags.extend([51, 29, 20, 42])
                if self.ra:
                    fillet_edges_tags.extend([33, 55, 26, 48])
                fillet_edges_tags.extend([53, 31, 32, 54, 24, 46, 52, 30, 44, 22])

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

    def _create_vertices(self) -> None:
        """ Create all vertices of the geometry. """
        # Make sure we have a graph to work with
        if not hasattr(self, "graph"):
            self.graph = nx.Graph()

        # Base vertices (always added)
        base_points = [1, 2, 3, 4, 5, 8, 9, 12, 15, 17, 19, 21, 27, 29, 31, 33]

        # Conditionally added vertices
        if self.la:
            base_points += [6, 7, 13, 25]
        if self.ra:
            base_points += [10, 11, 23, 35]
        if self.fl:
            if self.la:
                base_points += [14, 26]
            if self.ra:
                base_points += [24, 36]
            base_points += [16, 18, 20, 22, 28, 30, 32, 34]

        # Flange radii tags. Which are needed for explicit filleting
        self.flange_radii_tags = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]

        for tag in base_points:
            v = getattr(self, f"v{tag}")  # e.g. self.v1, self.v2...
            tag = self.model.occ.addPoint(*v, tag=tag)  # Add point in gmsh

            # Add a node to the graph with gmsh tag + coords
            self.graph.add_node(f"v{tag}", tag=tag, coords=tuple(v))

    def _create_edges(self) -> None:
        """ Create all edges of the geometry. """
        if not hasattr(self, "graph"):
            raise RuntimeError("Graph not initialized. Call _create_vertices first.")

        # Base edges
        edge_specs = [
            ("e1", 1, 2, 1),
            ("e2", 2, 3, 2),
            ("e3", 3, 4, 3),
            ("e4", 4, 1, 4),
            ("e5", 1, 5, 5),
            ("e6", 8, 5, 9),
            ("e7", 8, 4, 10),
            ("e8", 2, 9, 11),
            ("e9", 12, 9, 15),
            ("e10", 12, 3, 16),
            ("e11", 5, 15, 19),
            ("e12", 1, 17, 21),
            ("e13", 2, 19, 23),
            ("e14", 9, 21, 25),
            ("e15", 15, 17, 30),
            ("e16", 17, 19, 31),
            ("e17", 19, 21, 32),
            ("e18", 8, 27, 41),
            ("e19", 4, 29, 43),
            ("e20", 3, 31, 45),
            ("e21", 12, 33, 47),
            ("e22", 27, 29, 52),
            ("e23", 29, 31, 53),
            ("e24", 31, 33, 54),
        ]

        # Left addendum edges
        if self.la:
            edge_specs += [
                ("e25", 5, 6, 6),
                ("e26", 6, 7, 7),
                ("e27", 7, 8, 8),
                ("e28", 6, 13, 17),
                ("e29", 13, 15, 29),
                ("e30", 7, 25, 39),
                ("e31", 25, 27, 51),
            ]

        # Right addendum edges
        if self.ra:
            edge_specs += [
                ("e32", 9, 10, 12),
                ("e33", 10, 11, 13),
                ("e34", 11, 12, 14),
                ("e35", 10, 23, 27),
                ("e36", 21, 23, 33),
                ("e37", 11, 35, 49),
                ("e38", 33, 35, 55),
            ]

        # Flange edges
        if self.fl:
            if self.la:
                edge_specs += [
                    ("e39", 13, 14, 18),
                    ("e40", 14, 16, 34),
                    ("e41", 25, 26, 40),
                    ("e42", 26, 28, 56),
                ]
            if self.ra:
                edge_specs += [
                    ("e43", 23, 24, 28),
                    ("e44", 22, 24, 38),
                    ("e45", 35, 36, 50),
                    ("e46", 34, 36, 60),
                ]
            edge_specs += [
                ("e47", 15, 16, 20),
                ("e48", 17, 18, 22),
                ("e49", 19, 20, 24),
                ("e50", 21, 22, 26),
                ("e51", 16, 18, 35),
                ("e52", 18, 20, 36),
                ("e53", 20, 22, 37),
                ("e54", 27, 28, 42),
                ("e55", 29, 30, 44),
                ("e56", 31, 32, 46),
                ("e57", 33, 34, 48),
                ("e58", 28, 30, 57),
                ("e59", 30, 32, 58),
                ("e60", 32, 34, 59),
            ]

        # Create edges in gmsh and add them to the graph
        for edge_name, n1_tag, n2_tag, etag in edge_specs:
            tag = self.model.occ.addLine(n1_tag, n2_tag, tag=etag)
            self.graph.add_edge(f"v{n1_tag}", f"v{n2_tag}", name=edge_name, tag=tag, kind="line" )

        # For the explicit fillet method, mark the helper edges as those will be specifically removed later
        if self.fillet_method == "explicit":
            helper_tags = (1, 3, 5, 10, 11, 16, 21, 43, 23, 45)
            if self.fl:
                helper_tags += (30, 31, 32, 52, 53, 54, 22, 24, 44, 46)
            if self.la:
                helper_tags += (6, 8, 9, 19, 41)
                if self.fl:
                    helper_tags += (29, 51, 20, 42)
            if self.ra:
                helper_tags += (12, 14, 15, 25, 47)
                if self.fl:
                    helper_tags += (33, 55, 26, 48)
            # Set is_helper flag for helper edges in th graph
            for u, v, d in self.graph.edges(data=True):
                if d.get("tag") in helper_tags:
                    d["is_helper"] = True

    def _create_surfaces_procedural(self) -> None:
        """ Create all surfaces of the geometry using procedural construction """
        self.model.occ.addCurveLoop(curveTags=[4, 5, 9, 10], tag=3)
        self.model.occ.addSurfaceFilling(wireTag=3, tag=2, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop(curveTags=[1, 2, 3, 4], tag=5)
        self.model.occ.addSurfaceFilling(wireTag=5, tag=3, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop(curveTags=[2, 11, 15, 16], tag=7)
        self.model.occ.addSurfaceFilling(wireTag=7, tag=4, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop(curveTags=[5, 19, 30, 21], tag=13)
        self.model.occ.addSurfaceFilling(wireTag=13, tag=7, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop(curveTags=[1, 21, 31, 23], tag=15)
        self.model.occ.addSurfaceFilling(wireTag=15, tag=8, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop(curveTags=[11, 23, 32, 25], tag=17)
        self.model.occ.addSurfaceFilling(wireTag=17, tag=9, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop(curveTags=[10, 41, 52, 43], tag=23)
        self.model.occ.addSurfaceFilling(wireTag=23, tag=12, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop(curveTags=[3, 43, 53, 45], tag=25)
        self.model.occ.addSurfaceFilling(wireTag=25, tag=13, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop(curveTags=[16, 45, 54, 47], tag=27)
        self.model.occ.addSurfaceFilling(wireTag=27, tag=14, degree=2, maxDegree=3)

        if self.la:
            self.model.occ.addCurveLoop(curveTags=[6, 7, 8, 9], tag=1)
            self.model.occ.addSurfaceFilling(wireTag=1, tag=1, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[6, 17, 29, 19], tag=11)
            self.model.occ.addSurfaceFilling(wireTag=11, tag=6, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[8, 39, 51, 41], tag=21)
            self.model.occ.addSurfaceFilling(wireTag=21, tag=11, degree=2, maxDegree=3)

        if self.ra:
            self.model.occ.addCurveLoop(curveTags=[12, 13, 14, 15], tag=9)
            self.model.occ.addSurfaceFilling(wireTag=9, tag=5, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[12, 25, 33, 27], tag=19)
            self.model.occ.addSurfaceFilling(wireTag=19, tag=10, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[14, 47, 55, 49], tag=55)
            self.model.occ.addSurfaceFilling(wireTag=55, tag=15, degree=2, maxDegree=3)

        if self.fl:
            if self.la:
                self.model.occ.addCurveLoop(curveTags=[29, 18, 34, 20], tag=61)
                self.model.occ.addSurfaceFilling(wireTag=61, tag=16, degree=2, maxDegree=3)
                self.model.occ.addCurveLoop(curveTags=[40, 51, 42, 56], tag=41)
                self.model.occ.addSurfaceFilling(wireTag=41, tag=21, degree=2, maxDegree=3)

            if self.ra:
                self.model.occ.addCurveLoop(curveTags=[33, 26, 38, 28], tag=39)
                self.model.occ.addSurfaceFilling(wireTag=39, tag=20, degree=2, maxDegree=3)
                self.model.occ.addCurveLoop(curveTags=[48, 55, 50, 60], tag=49)
                self.model.occ.addSurfaceFilling(wireTag=49, tag=25, degree=2, maxDegree=3)

            self.model.occ.addCurveLoop(curveTags=[30, 20, 35, 22], tag=80)
            self.model.occ.addSurfaceFilling(wireTag=80, tag=17, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[31, 22, 36, 24], tag=35)
            self.model.occ.addSurfaceFilling(wireTag=35, tag=18, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[32, 24, 37, 26], tag=37)
            self.model.occ.addSurfaceFilling(wireTag=37, tag=19, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[42, 52, 44, 57], tag=43)
            self.model.occ.addSurfaceFilling(wireTag=43, tag=22, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[44, 53, 46, 58], tag=45)
            self.model.occ.addSurfaceFilling(wireTag=45, tag=23, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop(curveTags=[46, 54, 48, 59], tag=47)
            self.model.occ.addSurfaceFilling(wireTag=47, tag=24, degree=2, maxDegree=3)

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
        self._connect_fillet_nodes_on_yz_planes(tolerance=40)
        # Fillet the newly created circular arc junctions
        # Connect fillet edges with lines
        self._fillet_support_junctions()
        self._connect_all_junctions_with_bspline()
        self._connect_chain("bottom")
        self._connect_chain("top")

        self._debug_list_all_nodes()

        self._remove_helper_edges()
        self._fragment_at_graph_nodes()

        self._rebuild_graph_from_occ()

        self._clean_graph()
        surfaces = self._build_surfaces_from_ordered_cycles()
        self._build_shell_from_surfaces(surfaces)

if __name__ == '__main__':
    # Define directories and paths
    work_dir = r"path/to/your/working/directory"
    geom_path = Path(work_dir) / 'uchannel2.step'
    label_path = Path(work_dir) / 'uchannel2.json'

    # Instantiate the geometry
    u2 = UChannel2(geom_path, label_path, x_m=250, x_l=75, x_r=75, ll=100, lr=100, z_m=50, z_l=33.3, z_r=33.33,
                   y_m=52.5, y_l=56.25, y_r=56.25, y_fl=15, alpha_l=30, alpha_r=30, beta_l=15, beta_r=15, beta_m=15,
                   r=10, fl=True, la=True, ra=True)

    # Check the feasibility
    if u2.check_feasibility():
        # Create the geometry
        u2.fillet_method = 'explicit'
        u2.create_geometry(segmentation_labels=True, export=True, gui=True)
        u2.visualize_segmentation_graph()