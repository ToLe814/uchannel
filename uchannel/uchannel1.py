from pathlib import Path
from typing import Tuple
import gmsh
import math

from uchannel.uchannel_base import UChannelBase
from uchannel.error_handling import FeasibilityError


class UChannel1(UChannelBase):
    def __init__(self, path: Path, x_1: float = 100, x_2: float = 105, y_l: float = 50, y_r: float = 50,
                 y_sa: float = 50, z_l_1: float = 33.3, z_l_2: float = 33.3, z_r_1: float = 33.3, z_r_2: float = 33.3,
                 offset_mp_1: Tuple[float, float, float] = (1, 1, 1),
                 offset_mp_2: Tuple[float, float, float] = (1, 1, 1), r_l: Tuple[float, float] = (5, 5),
                 r_r: Tuple[float, float] = (5, 5), beta_l_1: float = 15, beta_l_2: float = 15, beta_r_1: float = 15,
                 beta_r_2: float = 15, sa: bool = 15):
        """
        Class for generating the UChannel1 geometry in gmsh. A short description of the parameters is given below.

        The key parameter for feasibility control is x_1. It can take any value. The authors recommend values between
        274 and 2477 mm. The remaining parameters are limited by x_1.

        :param path: Path to the .step file where the geometry will be exported
        :param x_1: Length of the bottom plane on side 1 along the x-axis
        :param x_2: Length of the bottom plane on side 2 along the x-axis
        :param y_l: Width of the bottom plane on left side along the y-axis
        :param y_r: Width of the bottom plane on right side along the y-axis
        :param y_sa: Width of the slant addendum along the y-axis
        :param z_l_1: Height of the left slant on side 1 along the z-axis
        :param z_l_2: Height of the left slant on side 2 along the z-axis
        :param z_r_1: Height of the right slant on side 1 along the z-axis
        :param z_r_2: Height of the right slant on side 2 along the z-axis
        :param offset_mp_1: Offset of the mid-point of the bottom plane (side 1) along the x, y, and z-axis
        :param offset_mp_2: Offset of the mid-point of the bottom plane (side 1) along the x, y, and z-axis
        :param r_l: Radii of the fillets on the left side
        :param r_r: Radii of the fillets on the right side
        :param beta_l_1: Angle of the left slant on side 1
        :param beta_l_2: Angle of the left slant on side 2
        :param beta_r_1: Angle of the right slant on side 1
        :param beta_r_2: Angle of the right slant on side 2
        :param sa: Boolean to indicate if the slant addendum should be created
        """
        super().__init__(path)

        # Parameters
        self.x_1 = x_1
        self.x_2 = x_2
        self.y_l = y_l
        self.y_r = y_r
        self.y_sa = y_sa
        self.z_l_1 = z_l_1
        self.z_l_2 = z_l_2
        self.z_r_1 = z_r_1
        self.z_r_2 = z_r_2
        self.offset_mp_1 = self.Vertex(*offset_mp_1)
        self.offset_mp_2 = self.Vertex(*offset_mp_2)
        self.r_l = r_l
        self.r_r = r_r
        self.beta_l_1 = math.radians(beta_l_1)
        self.beta_l_2 = math.radians(beta_l_2)
        self.beta_r_1 = math.radians(beta_r_1)
        self.beta_r_2 = math.radians(beta_r_2)
        self.sa = sa

        self.calculate_vertices()
        self.f_bounds = self._feasibility_bounds()  # Feasibility bounds

    def calculate_vertices(self) -> None:
        """
        Calculate the vertices of the UChannel1 geometry based on the parameters given in the constructor of UChannel1.
        The vertices order can be seen in the provided sketch.
        """
        # Vertices that construct the bottom plane
        self.v1 = self.Vertex(0, self.y_l, 0)
        self.v2 = self.Vertex(self.x_1, self.y_r, 0)
        self.v3 = self.Vertex(0, 0, 0)
        self.v4 = self.Vertex(self.x_2, 0, 0)
        self.v5 = self.Vertex((self.v2.x + self.v1.x) / 2 + self.offset_mp_1.x,
                              (self.v2.y + self.v1.y) / 2 + self.offset_mp_1.y, 0)
        self.v6 = self.Vertex((self.v3.x + self.v4.x) / 2 + self.offset_mp_2.x,
                              (self.v3.y + self.v4.y) / 2 + self.offset_mp_2.y, 0)

        # Vertices that construct the slants 1 and 2
        self.v7 = self.Vertex(0, self.y_l + self.z_l_1 * math.tan(self.beta_l_1), self.z_l_1)
        self.v8 = self.Vertex(self.x_1, self.y_r + self.z_r_1 * math.tan(self.beta_r_1), self.z_r_1)
        self.v9 = self.Vertex(0, -self.z_l_2 * math.tan(self.beta_l_2), self.z_l_2)
        self.v10 = self.Vertex(self.x_2, -self.z_r_2 * math.tan(self.beta_r_2), self.z_r_2)
        self.v11 = self.Vertex((self.v8.x + self.v7.x) / 2 + self.offset_mp_1.x,
                               (self.v8.y + self.v7.y) / 2 + self.offset_mp_1.y,
                               (self.v8.z + self.v7.z) / 2 + self.offset_mp_1.z)
        self.v12 = self.Vertex((self.v10.x + self.v9.x) / 2 + self.offset_mp_2.x,
                               (self.v10.y + self.v9.y) / 2 + self.offset_mp_2.y,
                               (self.v10.z + self.v9.z) / 2 + self.offset_mp_2.z)

        # Vertices that construct the slant addendum 1 and 2
        if self.sa:
            self.v13 = self.Vertex(self.v7.x, self.v7.y + self.y_sa, self.v7.z)
            self.v14 = self.Vertex(self.v8.x, self.v8.y + self.y_sa, self.v8.z)
            self.v15 = self.Vertex(self.v9.x, self.v9.y - self.y_sa, self.v9.z)
            self.v16 = self.Vertex(self.v10.x, self.v10.y - self.y_sa, self.v10.z)
            self.v17 = self.Vertex(self.v11.x, self.v11.y + self.y_sa, self.v11.z)
            self.v18 = self.Vertex(self.v12.x, self.v12.y - self.y_sa, self.v12.z)

    def _feasibility_bounds(self) -> dict:
        """
        Bounds for the feasibility check. Can be used as an input for an LHS sampler.
        """
        bounds = {
            "x_2": self.Bounds(0.95 * self.x_1, 1.05 * self.x_1),
            "y_l": self.Bounds(0.1 * self.x_1, 1 / 2 * self.x_1),
            "y_r": self.Bounds(0.1 * self.x_1, 1 / 2 * self.x_1),
            "y_sa": self.Bounds(1 / 40 * self.x_1, 1 / 2 * self.x_1),
            "z_l_1": self.Bounds(2 / 20 * self.x_1, 1 / 3 * self.x_1),
            "z_l_2": self.Bounds(2 / 20 * self.x_1, 1 / 3 * self.x_1),
            "z_r_1": self.Bounds(2 / 20 * self.x_1, 1 / 3 * self.x_1),
            "z_r_2": self.Bounds(2 / 20 * self.x_1, 1 / 3 * self.x_1),
            "offset_mp_1_x": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "offset_mp_1_y": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "offset_mp_1_z": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "offset_mp_2_x": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "offset_mp_2_y": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "offset_mp_2_z": self.Bounds(-0.01 * self.x_1, 0.01 * self.x_1),
            "r_l_0": self.Bounds(2 / 1000 * self.x_1, 1 / 20 * self.x_1),
            "r_l_1": self.Bounds(2 / 1000 * self.x_1, 1 / 20 * self.x_1),
            "r_r_0": self.Bounds(2 / 1000 * self.x_1, 1 / 20 * self.x_1),
            "r_r_1": self.Bounds(2 / 1000 * self.x_1, 1 / 20 * self.x_1),
            "beta_l_1": self.Bounds(math.radians(5), math.radians(15)),
            "beta_l_2": self.Bounds(math.radians(5), math.radians(15)),
            "beta_r_1": self.Bounds(math.radians(5), math.radians(15)),
            "beta_r_2": self.Bounds(math.radians(5), math.radians(15)),
            "sa": self.Bounds(True, False)
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
            self._validate_parameter("y_sa", self.y_sa, self.f_bounds["y_sa"].min, self.f_bounds["y_sa"].max)
            self._validate_parameter("z_l_1", self.z_l_1, self.f_bounds["z_l_1"].min, self.f_bounds["z_l_1"].max)
            self._validate_parameter("z_l_2", self.z_l_2, self.f_bounds["z_l_2"].min, self.f_bounds["z_l_2"].max)
            self._validate_parameter("z_r_1", self.z_r_1, self.f_bounds["z_r_1"].min, self.f_bounds["z_r_1"].max)
            self._validate_parameter("z_r_2", self.z_r_2, self.f_bounds["z_r_2"].min, self.f_bounds["z_r_2"].max)
            self._validate_parameter("offset_mp_1[0]", self.offset_mp_1.x, self.f_bounds["offset_mp_1_x"].min, self.f_bounds["offset_mp_1_x"].max)
            self._validate_parameter("offset_mp_1[1]", self.offset_mp_1.y, self.f_bounds["offset_mp_1_y"].min, self.f_bounds["offset_mp_1_y"].max)
            self._validate_parameter("offset_mp_1[2]", self.offset_mp_1.z, self.f_bounds["offset_mp_1_z"].min, self.f_bounds["offset_mp_1_z"].max)
            self._validate_parameter("offset_mp_2[0]", self.offset_mp_2.x, self.f_bounds["offset_mp_2_x"].min, self.f_bounds["offset_mp_2_x"].max)
            self._validate_parameter("offset_mp_2[1]", self.offset_mp_2.y, self.f_bounds["offset_mp_2_y"].min, self.f_bounds["offset_mp_2_y"].max)
            self._validate_parameter("offset_mp_2[2]", self.offset_mp_2.z, self.f_bounds["offset_mp_2_z"].min, self.f_bounds["offset_mp_2_z"].max)
            self._validate_parameter("y_sa", self.y_sa, self.f_bounds["y_sa"].min, self.f_bounds["y_sa"].max)
            self._validate_parameter("y_sa", self.y_sa, self.f_bounds["y_sa"].min, self.f_bounds["y_sa"].max)

            # Checks for geometry generation constraints (Overlaps, Intersections, Gmsh Feasibility etc.)
            self._validate_parameter("r_l[0]", self.r_l[0], self.f_bounds["r_l_0"].min, self.f_bounds["r_l_0"].max)
            self._validate_parameter("r_l[1]", self.r_l[1], self.f_bounds["r_l_1"].min, self.f_bounds["r_l_1"].max)
            self._validate_parameter("r_r[0]", self.r_r[0], self.f_bounds["r_r_0"].min, self.f_bounds["r_r_0"].max)
            self._validate_parameter("r_r[1]", self.r_r[1], self.f_bounds["r_r_1"].min, self.f_bounds["r_r_1"].max)

            # Checks for manufacturing constraints
            self._validate_parameter("beta_l_1", self.beta_l_1, self.f_bounds["beta_l_1"].min, self.f_bounds["beta_l_1"].max)
            self._validate_parameter("beta_l_2", self.beta_l_2, self.f_bounds["beta_l_2"].min, self.f_bounds["beta_l_2"].max)
            self._validate_parameter("beta_r_1", self.beta_r_1, self.f_bounds["beta_r_1"].min, self.f_bounds["beta_r_1"].max)
            self._validate_parameter("beta_r_2", self.beta_r_2, self.f_bounds["beta_r_2"].min, self.f_bounds["beta_r_2"].max)

        except FeasibilityError as e:
            print(f"FeasibilityError: {e}")
            return False

        return True

    def create_geometry(self, export: bool = True, gui: bool = False):
        """
        Create the UChannel1 geometry in gmsh using the parameters given in the constructor of UChannel1 and the
        calculated vertices.

        :param export: Export the geometry to the path given in the constructor of UChannel1
        :param gui: Launch the gmsh GUI

        :return: None
        """
        gmsh.initialize()
        self.model = gmsh.model

        # 1. Create vertices
        self._create_vertices()

        # 2. Create edges
        self._create_edges()

        # 3. Create surfaces
        self._create_surfaces()

        # 4. Create fillets
        if self.gmsh_filleting:
            if self.sa:
                fillet_edges_tags = [1, 2, 5, 6]
                self.create_fillet(
                    edges=fillet_edges_tags,
                    radii=[self.r_l[0], self.r_r[0], self.r_l[1], self.r_r[1], self.r_l[0], self.r_r[0], self.r_l[1],
                           self.r_r[1]]
                )
            else:
                fillet_edges_tags = [1, 2]
                self.create_fillet(
                    edges=fillet_edges_tags,
                    radii=[self.r_l[0], self.r_r[0], self.r_l[1], self.r_r[1]]
                )

        else:
            raise NotImplementedError("Gmsh uses the Open Cascade Kernel for filleting, whose filleting must not be "
                                      "feasible for all geometries. Users can implement their own filleting code here, "
                                      "if the provided filleting by the OCC is not sufficient.")

        # 5. Remove remaining non-geometry points (vertices)
        if self.sa:
            self.model.occ.remove([(0, 5), (0, 6), (0, 11), (0, 12), (0, 17), (0, 18)])
        else:
            self.model.occ.remove([(0, 5), (0, 6), (0, 11), (0, 12)])

        self.model.occ.synchronize()

        # Export the geometry
        if export:
            self._export_geometry()

        # Launch the GUI
        if gui:
            self._visualize_geometry()

        gmsh.finalize()

    def _create_vertices(self) -> None:
        # Bottom Plane
        self.model.occ.addPoint(*self.v1, tag=1)
        self.model.occ.addPoint(*self.v2, tag=2)
        self.model.occ.addPoint(*self.v3, tag=3)
        self.model.occ.addPoint(*self.v4, tag=4)

        # Slant 2
        self.model.occ.addPoint(*self.v5, tag=5)
        self.model.occ.addPoint(*self.v6, tag=6)
        self.model.occ.addPoint(*self.v7, tag=7)
        self.model.occ.addPoint(*self.v8, tag=8)

        # Slant 1
        self.model.occ.addPoint(*self.v9, tag=9)
        self.model.occ.addPoint(*self.v10, tag=10)
        self.model.occ.addPoint(*self.v11, tag=11)
        self.model.occ.addPoint(*self.v12, tag=12)

        # Slant Addendum
        if self.sa:
            self.model.occ.addPoint(*self.v13, tag=13)
            self.model.occ.addPoint(*self.v14, tag=14)
            self.model.occ.addPoint(*self.v15, tag=15)
            self.model.occ.addPoint(*self.v16, tag=16)
            self.model.occ.addPoint(*self.v17, tag=17)
            self.model.occ.addPoint(*self.v18, tag=18)

    def _create_edges(self) -> None:
        """
        Optionally instead of self.model.occ.addSpline one can use self.model.occ.addBSpline to create a B-spline curve.
        Pros:
            - The mid-point of the curve is not fixed in the x direction

        Cons:
            - The mid-point would not be on the curve. Rather a control point.
        """
        # Bottom Plane
        self.model.occ.addSpline([1, 5, 2], tag=1)
        self.model.occ.addSpline([3, 6, 4], tag=2)
        self.model.occ.addLine(1, 3, tag=3)
        self.model.occ.addLine(2, 4, tag=4)

        # Slant 1 and 2
        self.model.occ.addSpline([7, 11, 8], tag=5)
        self.model.occ.addSpline([9, 12, 10], tag=6)
        self.model.occ.addLine(1, 7, tag=7)
        self.model.occ.addLine(2, 8, tag=8)
        self.model.occ.addLine(3, 9, tag=9)
        self.model.occ.addLine(4, 10, tag=10)

        # Slant Addendum 1 and 2
        if self.sa:
            self.model.occ.addSpline([13, 17, 14], tag=11)
            self.model.occ.addLine(7, 13, tag=12)
            self.model.occ.addLine(8, 14, tag=13)
            self.model.occ.addLine(9, 15, tag=14)
            self.model.occ.addLine(10, 16, tag=15)
            self.model.occ.addSpline([15, 18, 16], tag=16)

    def _create_surfaces(self) -> None:

        # Bottom Plane
        self.model.occ.addCurveLoop([1, 3, 2, 4], 10)
        self.model.occ.addBSplineFilling(10, 1)

        # Slant 2
        self.model.occ.addCurveLoop([7, 1, 8, 5], 20)
        self.model.occ.addBSplineFilling(20, 2)

        # Slant 1
        self.model.occ.addCurveLoop([9, 2, 10, 6], 30)
        self.model.occ.addBSplineFilling(30, 3)

        if self.sa:
            # Slant Addendum 2
            self.model.occ.addCurveLoop([12, 5, 13, 11], 40)
            self.model.occ.addBSplineFilling(40, 4)

            # Slant Addendum 1
            self.model.occ.addCurveLoop([14, 16, 15, 6], 50)
            self.model.occ.addBSplineFilling(50, 5)


if __name__ == '__main__':
    # Define directories and paths
    work_dir = r"path/to/your/working/directory"
    geom_path = Path(work_dir) / 'uchannel1.step'

    # Instantiate the geometry
    u1 = UChannel1(geom_path, x_1=100, x_2=105, y_l=50, y_r=50, z_l_1=33.3, z_l_2=33.3, z_r_1=33.3, z_r_2=33.3,
                   offset_mp_1=(1, 1, 1), offset_mp_2=(1, 1, 1), r_l=(5, 5), r_r=(5, 5),
                   beta_l_1=15, beta_l_2=15, beta_r_1=15, beta_r_2=15, y_sa=50, sa=True)

    # Check the feasibility
    if u1.check_feasibility():
        # Create the geometry
        u1.gmsh_filleting = True
        u1.create_geometry(export=True, gui=True)
