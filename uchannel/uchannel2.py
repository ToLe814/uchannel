from pathlib import Path
import math

import gmsh

from uchannel.uchannel_base import UChannelBase
from uchannel.error_handling import FeasibilityError


class UChannel2(UChannelBase):
    def __init__(self, path: Path, x_m: float = 100, x_l: float = 25, x_r: float = 25, ll: float = 12.5,
                 lr: float = 12.5, z_m: float = 16.67, z_l: float = 11.12, z_r: float = 11.12, y_m: float = 20,
                 y_l: float = 10, y_r: float = 10, y_sa: float = 5, alpha_l: float = 0, alpha_r: float = 0,
                 beta_l: float = 5, beta_r: float = 5, beta_m: float = 5, r: float = 0.4, sa: bool = True,
                 la: bool = True, ra: bool = True):
        """
        Class for generating the UChannel2 geometry in gmsh. A short description of the parameters is given below.

        The key parameter for feasibility control is x_m. It can take any value. The authors recommend values between
        128 and 700 mm. The remaining parameters are limited by x_m.

        :param path: Path to the .step file where the geometry will be exported
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
        :param y_sa: Width of the slant in the y direction
        :param alpha_l: Angle of the left slanted plane
        :param alpha_r: Angle of the right slanted plane
        :param beta_l: Angle of the slant of the left plane in degrees
        :param beta_r: Angle of the slant of the right plane in degrees
        :param beta_m: Angle of the slant of the middle plane in degrees
        :param r: Radius of the fillets. Radius is the same for all fillets.
        :param sa: Boolean to determine if the slant addendum is present
        :param la: Boolean to determine if the left addendum is present
        :param ra: Boolean to determine if the right addendum is present
        """
        super().__init__(path)

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
        self.y_sa = y_sa
        self.alpha_l = math.radians(alpha_l)
        self.alpha_r = math.radians(alpha_r)
        self.beta_l = math.radians(beta_l)
        self.beta_r = math.radians(beta_r)
        self.beta_m = math.radians(beta_m)
        self.r = r
        self.sa = sa
        self.la = la
        self.ra = ra

        self.calculate_vertices()
        self.f_bounds = self._feasibility_bounds()  # Feasibility bounds

    def calculate_vertices(self) -> None:
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
        self.v14 = self.Vertex(self.v13.x, self.v13.y + self.y_sa, self.v13.z)
        self.v15 = self.Vertex(self.v13.x + self.x_l, self.v13.y, self.v13.z)
        self.v16 = self.Vertex(self.v15.x, self.v15.y + self.y_sa, self.v15.z)

        # Vertices that construct the slant of the middle plane m
        y_s_m = math.tan(self.beta_m) * self.z_m
        self.v17 = self.Vertex(self.v1.x, self.v1.y + y_s_m, self.v1.z + self.z_m)
        self.v18 = self.Vertex(self.v17.x, self.v17.y + self.y_sa, self.v17.z)
        self.v19 = self.Vertex(self.v17.x + self.x_m, self.v17.y, self.v17.z)
        self.v20 = self.Vertex(self.v19.x, self.v19.y + self.y_sa, self.v19.z)

        # Vertices that construct the slant of the right plane r
        y_s_r = math.tan(self.beta_r) * self.z_r
        self.v23 = self.Vertex(self.v10.x, self.v10.y + y_s_r, self.v10.z + self.z_r)
        self.v21 = self.Vertex(self.v23.x - self.x_r, self.v23.y, self.v23.z)
        self.v22 = self.Vertex(self.v21.x, self.v21.y + self.y_sa, self.v21.z)
        self.v24 = self.Vertex(self.v23.x, self.v23.y + self.y_sa, self.v23.z)

        # Vertices that construct the slant of the left plane l
        self.v25 = self.Vertex(self.v7.x, self.v7.y - y_s_l, self.v7.z + self.z_l)
        self.v26 = self.Vertex(self.v25.x, self.v25.y - self.y_sa, self.v25.z)
        self.v27 = self.Vertex(self.v25.x + self.x_l, self.v25.y, self.v25.z)
        self.v28 = self.Vertex(self.v27.x, self.v27.y - self.y_sa, self.v27.z)

        # Vertices that construct the slant of the middle plane m
        self.v29 = self.Vertex(self.v4.x, self.v4.y - y_s_m, self.v4.z + self.z_m)
        self.v30 = self.Vertex(self.v29.x, self.v29.y - self.y_sa, self.v29.z)
        self.v31 = self.Vertex(self.v29.x + self.x_m, self.v29.y, self.v29.z)
        self.v32 = self.Vertex(self.v31.x, self.v31.y - self.y_sa, self.v31.z)

        # Vertices that construct the slant of the right plane r
        self.v35 = self.Vertex(self.v11.x, self.v11.y - y_s_r, self.v11.z + self.z_r)
        self.v33 = self.Vertex(self.v35.x - self.x_r, self.v35.y, self.v35.z)
        self.v34 = self.Vertex(self.v33.x, self.v33.y - self.y_sa, self.v33.z)
        self.v36 = self.Vertex(self.v35.x, self.v35.y - self.y_sa, self.v35.z)

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
            "y_sa": self.Bounds(1/20*self.x_m, 3/20*self.x_m),
            "alpha_l": self.Bounds(0, math.radians(30)),
            "alpha_r": self.Bounds(0, math.radians(30)),
            "beta_l": self.Bounds(math.radians(5), math.radians(15)),
            "beta_r": self.Bounds(math.radians(5), math.radians(15)),
            "beta_m": self.Bounds(math.radians(5), math.radians(15)),
            "r": self.Bounds(1/250*self.x_m, 1/25*self.x_m),
            "sa": self.Bounds(True, False),
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
            if self.sa:
                self._validate_parameter("y_sa", self.y_sa, self.f_bounds["y_sa"].min, self.f_bounds["y_sa"].max)
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

    def create_geometry(self, export: bool = True, gui: bool = False) -> None:
        """
        Create the UChannel2 geometry in gmsh using the parameters given in the constructor of UChannel2 and the
        calculated vertices.

        :param export: Export the geometry to the path given in the constructor of UChannel2
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

        # 1.4 Create fillets
        if self.gmsh_filleting:
            # gather fillet edge tags
            fillet_edges_tags = [1, 2, 3, 4, 5, 10, 21, 43, 11, 16, 23, 45]  # Edges that need fillets weather sa, la,
            # ra or not

            if self.la:
                fillet_edges_tags.extend([6, 8, 19, 41, 9])

            if self.ra:
                fillet_edges_tags.extend([12, 14, 15, 25, 47])

            if self.sa:
                if self.la:
                    fillet_edges_tags.extend([51, 29, 20, 42])
                if self.ra:
                    fillet_edges_tags.extend([33, 55, 26, 48])
                fillet_edges_tags.extend([53, 31, 32, 54, 24, 46, 52, 30, 44, 22])

            # conduct filleting
            self.create_fillet(edges=fillet_edges_tags, radii=[self.r])

        else:
            raise NotImplementedError("Gmsh uses the Open Cascade Kernel for filleting, whose filleting must not be "
                                      "feasible for all geometries. Users can implement their own filleting code here, "
                                      "if the provided filleting by the OCC is not sufficient.")

        self.model.occ.synchronize()

        # Export the geometry
        if export:
            self._export_geometry()

        # Launch the GUI
        if gui:
            self._visualize_geometry()

        gmsh.finalize()

    def _create_vertices(self) -> None:
        self.model.occ.addPoint(*self.v1, tag=1)
        self.model.occ.addPoint(*self.v2, tag=2)
        self.model.occ.addPoint(*self.v3, tag=3)
        self.model.occ.addPoint(*self.v4, tag=4)
        self.model.occ.addPoint(*self.v5, tag=5)
        self.model.occ.addPoint(*self.v8, tag=8)
        self.model.occ.addPoint(*self.v9, tag=9)
        self.model.occ.addPoint(*self.v12, tag=12)
        self.model.occ.addPoint(*self.v15, tag=15)
        self.model.occ.addPoint(*self.v17, tag=17)
        self.model.occ.addPoint(*self.v19, tag=19)
        self.model.occ.addPoint(*self.v21, tag=21)
        self.model.occ.addPoint(*self.v27, tag=27)
        self.model.occ.addPoint(*self.v29, tag=29)
        self.model.occ.addPoint(*self.v31, tag=31)
        self.model.occ.addPoint(*self.v33, tag=33)

        if self.la:
            self.model.occ.addPoint(*self.v6, tag=6)
            self.model.occ.addPoint(*self.v7, tag=7)
            self.model.occ.addPoint(*self.v13, tag=13)
            self.model.occ.addPoint(*self.v25, tag=25)

        if self.ra:
            self.model.occ.addPoint(*self.v10, tag=10)
            self.model.occ.addPoint(*self.v11, tag=11)
            self.model.occ.addPoint(*self.v23, tag=23)
            self.model.occ.addPoint(*self.v35, tag=35)

        if self.sa:
            if self.la:
                self.model.occ.addPoint(*self.v14, tag=14)
                self.model.occ.addPoint(*self.v26, tag=26)

            if self.ra:
                self.model.occ.addPoint(*self.v24, tag=24)
                self.model.occ.addPoint(*self.v36, tag=36)

            self.model.occ.addPoint(*self.v16, tag=16)
            self.model.occ.addPoint(*self.v18, tag=18)
            self.model.occ.addPoint(*self.v20, tag=20)
            self.model.occ.addPoint(*self.v22, tag=22)
            self.model.occ.addPoint(*self.v28, tag=28)
            self.model.occ.addPoint(*self.v30, tag=30)
            self.model.occ.addPoint(*self.v32, tag=32)
            self.model.occ.addPoint(*self.v34, tag=34)

    def _create_edges(self) -> None:
        self.model.occ.addLine(1, 2, 1)
        self.model.occ.addLine(2, 3, 2)
        self.model.occ.addLine(3, 4, 3)
        self.model.occ.addLine(4, 1, 4)
        self.model.occ.addLine(1, 5, 5)
        self.model.occ.addLine(8, 5, 9)
        self.model.occ.addLine(8, 4, 10)
        self.model.occ.addLine(2, 9, 11)
        self.model.occ.addLine(12, 9, 15)
        self.model.occ.addLine(12, 3, 16)
        self.model.occ.addLine(5, 15, 19)
        self.model.occ.addLine(1, 17, 21)
        self.model.occ.addLine(2, 19, 23)
        self.model.occ.addLine(9, 21, 25)

        self.model.occ.addLine(15, 17, 30)
        self.model.occ.addLine(17, 19, 31)
        self.model.occ.addLine(19, 21, 32)
        self.model.occ.addLine(8, 27, 41)
        self.model.occ.addLine(4, 29, 43)
        self.model.occ.addLine(3, 31, 45)
        self.model.occ.addLine(12, 33, 47)
        self.model.occ.addLine(27, 29, 52)
        self.model.occ.addLine(29, 31, 53)
        self.model.occ.addLine(31, 33, 54)

        if self.la:
            self.model.occ.addLine(5, 6, 6)
            self.model.occ.addLine(6, 7, 7)
            self.model.occ.addLine(7, 8, 8)
            self.model.occ.addLine(6, 13, 17)
            self.model.occ.addLine(13, 15, 29)
            self.model.occ.addLine(7, 25, 39)
            self.model.occ.addLine(25, 27, 51)

        if self.ra:
            self.model.occ.addLine(9, 10, 12)
            self.model.occ.addLine(10, 11, 13)
            self.model.occ.addLine(11, 12, 14)
            self.model.occ.addLine(10, 23, 27)
            self.model.occ.addLine(21, 23, 33)
            self.model.occ.addLine(11, 35, 49)
            self.model.occ.addLine(33, 35, 55)

        if self.sa:
            if self.la:
                self.model.occ.addLine(13, 14, 18)
                self.model.occ.addLine(14, 16, 34)
                self.model.occ.addLine(25, 26, 40)
                self.model.occ.addLine(26, 28, 56)

            if self.ra:
                self.model.occ.addLine(23, 24, 28)
                self.model.occ.addLine(22, 24, 38)
                self.model.occ.addLine(35, 36, 50)
                self.model.occ.addLine(34, 36, 60)

            self.model.occ.addLine(15, 16, 20)
            self.model.occ.addLine(17, 18, 22)
            self.model.occ.addLine(19, 20, 24)
            self.model.occ.addLine(21, 22, 26)
            self.model.occ.addLine(16, 18, 35)
            self.model.occ.addLine(18, 20, 36)
            self.model.occ.addLine(20, 22, 37)
            self.model.occ.addLine(27, 28, 42)
            self.model.occ.addLine(29, 30, 44)
            self.model.occ.addLine(31, 32, 46)
            self.model.occ.addLine(33, 34, 48)
            self.model.occ.addLine(28, 30, 57)
            self.model.occ.addLine(30, 32, 58)
            self.model.occ.addLine(32, 34, 59)

    def _create_faces(self) -> None:
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

        if self.sa:
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


if __name__ == '__main__':
    # Define directories and paths
    work_dir = r"path/to/your/working/directory"
    geom_path = Path(work_dir) / 'uchannel2.step'

    # Instantiate the geometry
    u2 = UChannel2(geom_path, x_m=250, x_l=75, x_r=75, ll=100, lr=100, z_m=50, z_l=33.3, z_r=33.33, y_m=52.5, y_l=56.25,
                   y_r=56.25, y_sa=15, alpha_l=30, alpha_r=30, beta_l=15, beta_r=15, beta_m=15, r=0.0, sa=True,
                   la=True, ra=True)

    # Check the feasibility
    if u2.check_feasibility():
        # Create the geometry
        u2.gmsh_filleting = True
        u2.create_geometry(export=True, gui=True)
