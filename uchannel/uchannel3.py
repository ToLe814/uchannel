from pathlib import Path
import math

import gmsh

from uchannel.uchannel_base import UChannelBase
from uchannel.error_handling import FeasibilityError


class UChannel3(UChannelBase):
    def __init__(self,
                 path: Path, label_path: Path, x_b1: float = 100, x_b2: float = 60, x_ad: float = 40, l_c1: float = 40, l_c2: float = 40,
                 x_t: float = 60, y_m: float = 50, alpha: float = 20, z_ad: float = 10, z_b: float = 20,
                 z_t: float = 20, beta_b: float = 15, beta_c: float = 15, beta_t: float = 15, y_sa: float = 10,
                 r: float = 5, ad: bool = True, sa: bool = False):
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
        :param y_sa: Width of the slant addendum in y direction
        :param r: Radius of the fillets
        :param ad: Boolean to decide if the slant addendum should be added
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
        self.y_sa = y_sa
        self.z_ad = z_ad
        self.z_b = z_b
        self.z_t = z_t
        self.alpha = math.radians(alpha)
        self.beta_b = math.radians(beta_b)
        self.beta_c = math.radians(beta_c)
        self.beta_t = math.radians(beta_t)
        self.r = r
        self.ad = ad
        self.sa = sa

        self.calculate_vertices()
        self.f_bounds = self._feasibility_bounds()  # Feasibility bounds

    def calculate_vertices(self):
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

        # Vertices that construct the s1 slant addendum
        self.v34 = self.Vertex(self.v29.x, self.v29.y + self.y_sa, self.v29.z)
        self.v33 = self.Vertex(self.v25.x, self.v25.y + self.y_sa, self.v25.z)
        self.v19 = self.Vertex(self.v20.x, self.v20.y + self.y_sa, self.v20.z)
        self.v13 = self.Vertex(self.v14.x, self.v14.y + self.y_sa, self.v14.z)
        self.v7 = self.Vertex(self.v8.x, self.v8.y + self.y_sa, self.v8.z)
        self.v1 = self.Vertex(self.v2.x, self.v2.y + self.y_sa, self.v2.z)
        self.v37 = self.Vertex(self.v38.x, self.v38.y + self.y_sa, self.v38.z)

        # Vertices that construct the s2 slant and addendum
        self.v11 = self.Vertex(self.v10.x, self.v10.y - y_s1_b, self.v10.z + self.z_b)
        self.v5 = self.Vertex(self.v11.x - self.x_b1, self.v11.y, self.v11.z)
        self.v41 = self.Vertex(self.v5.x - self.x_ad, self.v5.y, self.v5.z)
        self.v32 = self.Vertex(self.v31.x, self.v31.y - y_s1_t, self.v31.z + self.z_t)
        self.v28 = self.Vertex(self.v32.x - self.x_t, self.v32.y, self.v32.z)
        self.v23 = self.Vertex(self.v28.x - self.l_c2 * math.cos(self.alpha), self.v31.y - y_s1_c, self.v28.z)
        self.v17 = self.Vertex(self.v23.x - self.l_c1 * math.cos(self.alpha), self.v23.y,
                               self.z_b + ((self.v15.z - self.v9.x) / (self.v20.x - self.v9.x)) * (self.v20.z - self.v8.z))
        self.v36 = self.Vertex(self.v32.x, self.v32.y - self.y_sa, self.v32.z)
        self.v35 = self.Vertex(self.v28.x, self.v28.y - self.y_sa, self.v28.z)
        self.v24 = self.Vertex(self.v23.x, self.v23.y - self.y_sa, self.v23.z)
        self.v18 = self.Vertex(self.v17.x, self.v17.y - self.y_sa, self.v17.z)
        self.v12 = self.Vertex(self.v11.x, self.v11.y - self.y_sa, self.v11.z)
        self.v6 = self.Vertex(self.v5.x, self.v5.y - self.y_sa, self.v5.z)
        self.v42 = self.Vertex(self.v41.x, self.v41.y - self.y_sa, self.v41.z)

        # Vertices that construct the ad plane
        self.v39 = self.Vertex(-self.x_b2 - self.x_b1 - self.x_ad, 0., self.z_ad)
        self.v40 = self.Vertex(-self.x_b2 - self.x_b1 - self.x_ad, -self.y_m, self.z_ad)

        # Helper points for spline in ad
        self.hp1 = self.Vertex(self.v3.x - 0.25 * self.x_ad, self.v3.y, self.v3.z + 0.1 * self.z_ad)
        self.hp2 = self.Vertex(self.v3.x - 0.50 * self.x_ad, self.v3.y, self.v3.z + 0.5 * self.z_ad)
        self.hp3 = self.Vertex(self.v3.x - 0.75 * self.x_ad, self.v3.y, self.v3.z + 0.9 * self.z_ad)
        self.hp4 = self.Vertex(self.hp1.x, self.hp1.y - self.y_m, self.hp1.z)
        self.hp5 = self.Vertex(self.hp2.x, self.hp2.y - self.y_m, self.hp2.z)
        self.hp6 = self.Vertex(self.hp3.x, self.hp3.y - self.y_m, self.hp3.z)

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
            "y_sa": self.Bounds(self.x_b1 * 1 / 20, self.x_b1 * 1 / 10),
            "r": self.Bounds(self.x_b1 * 1 / 100, self.x_b1 * 1 / 30),
            "ad": self.Bounds(True, False),
            "sa": self.Bounds(True, False)
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
            self._validate_parameter('y_sa', self.y_sa, self.f_bounds["y_sa"].min, self.f_bounds["y_sa"].max)
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
        gmsh.initialize()
        self.model = gmsh.model
        self.model.add(self.name)

        # 1.1 Create vertices calculated in self.calculate_vertices()
        self._create_vertices()

        # 1.2 create edges based on the vertices
        self._create_edges()

        # 1.3 create faces based on the edges
        self._create_faces()

        # 1.4 create fillets
        if self.gmsh_filleting:
            # gather fillet edges
            fillet_edges_tags = [6, 7, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41,
                                 42, 43, 44, 45, 46, 47]
            if self.sa:
                fillet_edges_tags.extend([10, 6, 16, 20, 33, 34, 48, 49])
            if self.ad:
                fillet_edges_tags.extend([62, 63, 64, 65])

            # conduct filleting
            self.create_fillet(edges=fillet_edges_tags, radii=[self.r])

        else:
            raise NotImplementedError("Gmsh uses the Open Cascade Kernel for filleting, whose filleting must not be "
                                      "feasible for all geometries. Users can implement their own filleting code here, "
                                      "if the provided filleting by the OCC is not sufficient.")

        if self.ad:
            # remove helper points
            self.model.occ.remove([(0, 101), (0, 102), (0, 103), (0, 104), (0, 105), (0, 106)])

        self.model.occ.synchronize()

        # Get the segmentation labels (auto-detects fillet + flange settings)
        if segmentation_labels:
            self.graph = self.face_adjacency.create_graph()
            self.get_segmentation_labels()

        # Export the geometry
        if export:
            self._export_geometry()
            if segmentation_labels:
                self._write_segmentation_labels()

        # Launch the GUI
        if gui:
            self._visualize_geometry()

        gmsh.finalize()

    def _create_vertices(self):
        self.model.occ.addPoint(*self.v1, 0, 1)
        self.model.occ.addPoint(*self.v2, 0, 2)
        self.model.occ.addPoint(*self.v3, 0, 3)
        self.model.occ.addPoint(*self.v4, 0, 4)
        self.model.occ.addPoint(*self.v5, 0, 5)
        self.model.occ.addPoint(*self.v6, 0, 6)
        self.model.occ.addPoint(*self.v7, 0, 7)
        self.model.occ.addPoint(*self.v8, 0, 8)
        self.model.occ.addPoint(*self.v9, 0, 9)
        self.model.occ.addPoint(*self.v10, 0, 10)
        self.model.occ.addPoint(*self.v11, 0, 11)
        self.model.occ.addPoint(*self.v12, 0, 12)
        self.model.occ.addPoint(*self.v13, 0, 13)
        self.model.occ.addPoint(*self.v15, 0, 15)
        self.model.occ.addPoint(*self.v17, 0, 17)
        self.model.occ.addPoint(*self.v19, 0, 19)
        self.model.occ.addPoint(*self.v21, 0, 21)
        self.model.occ.addPoint(*self.v23, 0, 23)
        self.model.occ.addPoint(*self.v25, 0, 25)
        self.model.occ.addPoint(*self.v27, 0, 27)
        self.model.occ.addPoint(*self.v29, 0, 29)
        self.model.occ.addPoint(*self.v31, 0, 31)
        self.model.occ.addPoint(*self.v14, 0, 14)
        self.model.occ.addPoint(*self.v16, 0, 16)
        self.model.occ.addPoint(*self.v18, 0, 18)
        self.model.occ.addPoint(*self.v20, 0, 20)
        self.model.occ.addPoint(*self.v22, 0, 22)
        self.model.occ.addPoint(*self.v24, 0, 24)
        self.model.occ.addPoint(*self.v26, 0, 26)
        self.model.occ.addPoint(*self.v28, 0, 28)
        self.model.occ.addPoint(*self.v30, 0, 30)
        self.model.occ.addPoint(*self.v32, 0, 32)

        if self.sa:
            self.model.occ.addPoint(*self.v33, 0, 33)
            self.model.occ.addPoint(*self.v34, 0, 34)
            self.model.occ.addPoint(*self.v35, 0, 35)
            self.model.occ.addPoint(*self.v36, 0, 36)

        if self.ad:
            self.model.occ.addPoint(*self.v37, 0, 37)
            self.model.occ.addPoint(*self.v38, 0, 38)
            self.model.occ.addPoint(*self.v39, 0, 39)
            self.model.occ.addPoint(*self.v40, 0, 40)
            self.model.occ.addPoint(*self.v41, 0, 41)
            self.model.occ.addPoint(*self.v42, 0, 42)

            # helper points
            # edge 63
            self.model.occ.addPoint(*self.hp1, 0, 101)
            self.model.occ.addPoint(*self.hp2, 0, 102)
            self.model.occ.addPoint(*self.hp3, 0, 103)

            # edge 64
            self.model.occ.addPoint(*self.hp4, 0, 104)
            self.model.occ.addPoint(*self.hp5, 0, 105)
            self.model.occ.addPoint(*self.hp6, 0, 106)

    def _create_edges(self):
        self.model.occ.addLine(1, 2, 1)
        self.model.occ.addLine(2, 3, 2)
        self.model.occ.addLine(3, 4, 3)
        self.model.occ.addLine(4, 5, 4)
        self.model.occ.addLine(5, 6, 5)
        self.model.occ.addLine(7, 8, 6)
        self.model.occ.addLine(8, 9, 7)
        self.model.occ.addLine(9, 10, 8)
        self.model.occ.addLine(10, 11, 9)
        self.model.occ.addLine(11, 12, 10)
        self.model.occ.addLine(13, 14, 11)
        self.model.occ.addLine(14, 15, 12)
        self.model.occ.addLine(15, 16, 13)
        self.model.occ.addLine(16, 17, 14)
        self.model.occ.addLine(17, 18, 15)
        self.model.occ.addLine(19, 20, 16)
        self.model.occ.addLine(20, 21, 17)
        self.model.occ.addLine(21, 22, 18)
        self.model.occ.addLine(22, 23, 19)
        self.model.occ.addLine(23, 24, 20)
        self.model.occ.addLine(25, 26, 21)
        self.model.occ.addLine(26, 27, 22)
        self.model.occ.addLine(27, 28, 23)
        self.model.occ.addLine(29, 30, 24)
        self.model.occ.addLine(30, 31, 25)
        self.model.occ.addLine(31, 32, 26)
        self.model.occ.addLine(1, 7, 27)
        self.model.occ.addLine(7, 13, 28)
        self.model.occ.addLine(13, 19, 29)
        self.model.occ.addLine(2, 8, 30)
        self.model.occ.addLine(8, 14, 31)
        self.model.occ.addLine(14, 20, 32)
        self.model.occ.addLine(20, 25, 33)
        self.model.occ.addLine(25, 29, 34)
        self.model.occ.addLine(3, 9, 35)
        self.model.occ.addLine(9, 15, 36)
        self.model.occ.addLine(15, 21, 37)
        self.model.occ.addLine(21, 26, 38)
        self.model.occ.addLine(26, 30, 39)
        self.model.occ.addLine(4, 10, 40)
        self.model.occ.addLine(10, 16, 41)
        self.model.occ.addLine(16, 22, 42)
        self.model.occ.addLine(22, 27, 43)
        self.model.occ.addLine(27, 31, 44)
        self.model.occ.addLine(5, 11, 45)
        self.model.occ.addLine(11, 17, 46)
        self.model.occ.addLine(17, 23, 47)
        self.model.occ.addLine(23, 28, 48)
        self.model.occ.addLine(28, 32, 49)
        self.model.occ.addLine(6, 12, 50)
        self.model.occ.addLine(12, 18, 51)
        self.model.occ.addLine(18, 24, 52)

        if self.sa:
            self.model.occ.addLine(19, 33, 53)
            self.model.occ.addLine(33, 34, 54)
            self.model.occ.addLine(33, 25, 55)
            self.model.occ.addLine(34, 29, 56)
            self.model.occ.addLine(24, 35, 57)
            self.model.occ.addLine(35, 36, 58)
            self.model.occ.addLine(35, 28, 59)
            self.model.occ.addLine(36, 32, 60)

        if self.ad:
            self.model.occ.addLine(1, 37, 61)
            self.model.occ.addLine(2, 38, 62)
            self.model.occ.addSpline([3, 101, 102, 103, 39], 63)
            self.model.occ.addSpline([4, 104, 105, 106, 40], 64)
            self.model.occ.addLine(5, 41, 65)
            self.model.occ.addLine(6, 42, 66)
            self.model.occ.addLine(37, 38, 67)
            self.model.occ.addLine(38, 39, 68)
            self.model.occ.addLine(39, 40, 69)
            self.model.occ.addLine(40, 41, 70)
            self.model.occ.addLine(41, 42, 71)

    def _create_faces(self):
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

        if self.sa:
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


if __name__ == '__main__':
    # Define directories and paths
    work_dir = r"path/to/your/working/directory"
    geom_path = Path(work_dir) / 'uchannel3.step'
    label_path = Path(work_dir) / 'uchannel3.json'

    # Instantiate the geometry
    u3 = UChannel3(geom_path, label_path, x_b1=100, x_b2=10, x_ad=20, l_c1=14.29, l_c2=14.92, x_t=10, y_m=20, alpha=20, z_ad=0,
                   z_b=10, z_t=6.7, beta_b=5, beta_c=15, beta_t=5, y_sa=5, r=3.0, ad=True, sa=True)

    # Check the feasibility
    if u3.check_feasibility():
        # Create the geometry
        u3.gmsh_filleting = True
        u3.create_geometry(segmentation_labels=True, export=True, gui=False)
        u3.visualize_segmentation_graph()