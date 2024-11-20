from pathlib import Path
import math

import gmsh

from uchannel.uchannel_base import UChannelBase
from uchannel.error_handling import FeasibilityError


class UChannel4(UChannelBase):
    def __init__(self, path: Path, x_m: float = 100, ll: float = 50, lr: float = 50, z_m: float = 40, z_l: float = 53.3,
                 z_r: float = 53.3, y_m: float = 50, y_l: float = 62.5, y_r: float = 62.5, y_sa: float = 20,
                 r: float = 12.5, sa: bool = True, alpha_l: float = 25, alpha_r: float = 25, beta_m: float = 25,
                 beta_l: float = 15, beta_r: float = 15):
        """
        Class for generating the UChannel4 geometry in gmsh. A short description of the parameters is given below.

        The key parameter for feasibility control is x_m. It can take any value. The authors recommend values between
        189 and 1223 mm. The remaining parameters are limited by x_m.

        :param path: Path to the .step file where the geometry will be exported
        :param x_m: Length of the middle plane m in x direction
        :param ll: Length of the left plane l in normal direction
        :param lr: Length of the right plane r in normal direction
        :param z_m: Height of the middle plane m in z direction
        :param z_l: Height of the left plane l in z direction
        :param z_r: Height of the right plane r in z direction
        :param y_m: Width of the middle plane m in y direction
        :param y_l: Width of the left plane l in y direction
        :param y_r: Width of the right plane r in y direction
        :param y_sa: Width of the slant addendum in y direction
        :param r: Radius of the fillets. Same for all fillets.
        :param sa: Boolean to decide if the slant addendum should be added
        :param alpha_l: Angle of the left plane l from the x-axis in degrees
        :param alpha_r: Angle of the right plane r from the x-axis in degrees
        :param beta_m: Angle of the slant (middle plane) in degrees
        :param beta_l: Angle of the slant (left plane) in degrees
        :param beta_r: Angle of the slant (right plane) in degrees
        """
        super().__init__(path)

        # Parameters
        self.x_m = x_m
        self.ll = ll
        self.lr = lr
        self.z_m = z_m
        self.z_l = z_l
        self.z_r = z_r
        self.y_m = y_m
        self.y_l = y_l
        self.y_r = y_r
        self.y_sa = y_sa
        self.r = r
        self.sa = sa
        self.alpha_l = math.radians(alpha_l)
        self.alpha_r = math.radians(alpha_r)
        self.beta_m = math.radians(beta_m)
        self.beta_l = math.radians(beta_l)
        self.beta_r = math.radians(beta_r)

        self.calculate_vertices()
        self.f_bounds = self._feasibility_bounds()  # Feasibility bounds

    def calculate_vertices(self) -> None:
        """
        Calculate the vertices of the UChannel4 geometry based on the parameters given in the constructor of UChannel4.
        The vertices order can be seen in the provided sketch.
        """
        # Vertices that construct the middle plane m
        self.v6 = self.Vertex(0., 0., 0.)
        self.v7 = self.Vertex(0., -self.y_m, 0.)
        self.v10 = self.Vertex(self.x_m, 0., 0.)
        self.v11 = self.Vertex(self.x_m, -self.y_m, 0.)

        # Vertices that construct the left plane l
        self.v2 = self.Vertex(-(math.cos(self.alpha_l)*self.ll),
                              (self.y_l-self.y_m)/2,
                              -(math.sin(self.alpha_l) * self.ll))
        self.v3 = self.Vertex(-(math.cos(self.alpha_l)*self.ll),
                              -(self.y_l-self.y_m)/2-self.y_m,
                              -(math.sin(self.alpha_l) * self.ll))

        # Vertices that construct the right plane r
        self.v14 = self.Vertex((math.cos(self.alpha_r) * self.lr) + self.x_m,
                               (self.y_r-self.y_m)/2,
                               -(math.sin(self.alpha_r) * self.lr))
        self.v15 = self.Vertex((math.cos(self.alpha_r) * self.lr) + self.x_m,
                               -(self.y_r-self.y_m)/2-self.y_m,
                               -(math.sin(self.alpha_r) * self.lr))

        # Vertices that construct the left slant
        y_s = self.z_m * math.tan(self.beta_m)
        y_s_l = self.z_l * math.tan(self.beta_l)
        y_s_r = self.z_r * math.tan(self.beta_r)

        self.v1 = self.Vertex(self.v2.x, self.v2.y + y_s_l, self.v2.z + self.z_l)
        self.v5 = self.Vertex(self.v6.x, self.v6.y + y_s, self.v6.z + self.z_m)
        self.v9 = self.Vertex(self.v10.x, self.v10.y + y_s, self.v10.z + self.z_m)
        self.v13 = self.Vertex(self.v14.x, self.v14.y + y_s_r, self.v14.z + self.z_r)

        # Vertices that construct the left slant addendum
        self.v17 = self.Vertex(self.v1.x, self.v1.y + self.y_sa, self.v1.z)
        self.v18 = self.Vertex(self.v5.x, self.v5.y + self.y_sa, self.v5.z)
        self.v19 = self.Vertex(self.v9.x, self.v9.y + self.y_sa, self.v9.z)
        self.v20 = self.Vertex(self.v13.x, self.v13.y + self.y_sa, self.v13.z)

        # Vertices that construct the right slant
        self.v4 = self.Vertex(self.v3.x, self.v3.y - y_s_l, self.v3.z + self.z_l)
        self.v8 = self.Vertex(self.v7.x, self.v7.y - y_s, self.v7.z + self.z_m)
        self.v12 = self.Vertex(self.v11.x, self.v11.y - y_s, self.v11.z + self.z_m)
        self.v16 = self.Vertex(self.v15.x, self.v15.y - y_s_r, self.v15.z + self.z_r)

        # Vertices that construct the right slant addendum
        self.v21 = self.Vertex(self.v4.x, self.v4.y - self.y_sa, self.v4.z)
        self.v22 = self.Vertex(self.v8.x, self.v8.y - self.y_sa, self.v8.z)
        self.v23 = self.Vertex(self.v12.x, self.v12.y - self.y_sa, self.v12.z)
        self.v24 = self.Vertex(self.v16.x, self.v16.y - self.y_sa, self.v16.z)

    def _feasibility_bounds(self) -> dict:
        """
        Bounds for the feasibility check. Can be used as an input for an LHS sampler.
        """
        bounds = {
            "ll": self.Bounds(1/4*self.x_m, 1/2*self.x_m),
            "lr": self.Bounds(1/4*self.x_m, 1/2*self.x_m),
            "z_m": self.Bounds(1/5*self.x_m, 3/5*self.x_m),
            "z_l": self.Bounds(3/4*self.z_m, 4/3*self.z_m),
            "z_r": self.Bounds(3/4*self.z_m, 4/3*self.z_m),
            "y_m": self.Bounds(1/4*self.x_m, 1/2*self.x_m),
            "y_l": self.Bounds(1/4*self.x_m, 3/8*self.x_m),
            "y_r": self.Bounds(1/4*self.x_m, 3/8*self.x_m),
            "y_sa": self.Bounds(1/8*self.x_m, 1/5*self.x_m),
            "r": self.Bounds(1/150 * self.x_m, 1/11 * self.x_m),
            "sa": self.Bounds(True, False),
            "alpha_l": self.Bounds(0, math.radians(25)),
            "alpha_r": self.Bounds(0, math.radians(25)),
            "beta_m": self.Bounds(math.radians(5), math.radians(15)),
            "beta_l": self.Bounds(math.radians(5), math.radians(15)),
            "beta_r": self.Bounds(math.radians(5), math.radians(15))
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
            self._validate_parameter('ll', self.ll, self.f_bounds['ll'].min, self.f_bounds['ll'].max)
            self._validate_parameter('lr', self.lr, self.f_bounds['lr'].min, self.f_bounds['lr'].max)
            self._validate_parameter('z_m', self.z_m, self.f_bounds['z_m'].min, self.f_bounds['z_m'].max)
            self._validate_parameter('z_l', self.z_l, self.f_bounds['z_l'].min, self.f_bounds['z_l'].max)
            self._validate_parameter('z_r', self.z_r, self.f_bounds['z_r'].min, self.f_bounds['z_r'].max)
            self._validate_parameter('y_m', self.y_m, self.f_bounds['y_m'].min, self.f_bounds['y_m'].max)
            self._validate_parameter('y_l', self.y_l, self.f_bounds['y_l'].min, self.f_bounds['y_l'].max)
            self._validate_parameter('y_r', self.y_r, self.f_bounds['y_r'].min, self.f_bounds['y_r'].max)
            self._validate_parameter('y_sa', self.y_sa, self.f_bounds['y_sa'].min, self.f_bounds['y_sa'].max)
            self._validate_parameter('alpha_l', self.alpha_l, self.f_bounds['alpha_l'].min, self.f_bounds['alpha_l'].max)
            self._validate_parameter('alpha_r', self.alpha_r, self.f_bounds['alpha_r'].min, self.f_bounds['alpha_r'].max)

            # Checks for geometry generation constraints (Overlaps, Intersections, Gmsh Feasibility etc.)
            self._validate_parameter('r', self.r, self.f_bounds['r'].min, self.f_bounds['r'].max)

            # Checks for manufacturing constraints
            self._validate_parameter('beta_m', self.beta_m, self.f_bounds['beta_m'].min, self.f_bounds['beta_m'].max)
            self._validate_parameter('beta_l', self.beta_l, self.f_bounds['beta_l'].min, self.f_bounds['beta_l'].max)
            self._validate_parameter('beta_r', self.beta_r, self.f_bounds['beta_r'].min, self.f_bounds['beta_r'].max)

        except FeasibilityError as e:
            print(f"FeasibilityError: {e}")
            return False

        return True

    def create_geometry(self, export: bool = True, gui: bool = False) -> None:
        """
        Create the UChannel4 geometry in gmsh using the parameters given in the constructor of UChannel4 and the
        calculated vertices.

        :param export: Export the geometry to the path given in the constructor of UChannel4
        :param gui: Launch the gmsh GUI

        :return: None
        """
        gmsh.initialize()
        self.model = gmsh.model
        self.model.add(self.name)

        # 1.1 Create vertices calculated in self.calculate_vertices()
        self._create_vertices()

        # 1.2 Create edges
        self._create_edges()

        # 1.3 Create faces
        self._create_faces()

        # 1.4 Create fillets
        if self.gmsh_filleting:
            # gather fillet edges
            fillet_edges_tags = [4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21]
            if self.sa:
                fillet_edges_tags.extend([26, 27, 33, 34, 13, 14, 15, 22, 23, 24])

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
        self.model.occ.addPoint(*self.v6, tag=6)
        self.model.occ.addPoint(*self.v7, tag=7)
        self.model.occ.addPoint(*self.v8, tag=8)
        self.model.occ.addPoint(*self.v9, tag=9)
        self.model.occ.addPoint(*self.v10, tag=10)
        self.model.occ.addPoint(*self.v11, tag=11)
        self.model.occ.addPoint(*self.v12, tag=12)
        self.model.occ.addPoint(*self.v13, tag=13)
        self.model.occ.addPoint(*self.v14, tag=14)
        self.model.occ.addPoint(*self.v15, tag=15)
        self.model.occ.addPoint(*self.v16, tag=16)
        if self.sa:
            self.model.occ.addPoint(*self.v17, tag=17)
            self.model.occ.addPoint(*self.v18, tag=18)
            self.model.occ.addPoint(*self.v19, tag=19)
            self.model.occ.addPoint(*self.v20, tag=20)
            self.model.occ.addPoint(*self.v21, tag=21)
            self.model.occ.addPoint(*self.v22, tag=22)
            self.model.occ.addPoint(*self.v23, tag=23)
            self.model.occ.addPoint(*self.v24, tag=24)

    def _create_edges(self) -> None:
        self.model.occ.addLine(1, 2, tag=1)
        self.model.occ.addLine(2, 3, tag=2)
        self.model.occ.addLine(3, 4, tag=3)
        self.model.occ.addLine(5, 6, tag=4)
        self.model.occ.addLine(6, 7, tag=5)
        self.model.occ.addLine(7, 8, tag=6)
        self.model.occ.addLine(9, 10, tag=7)
        self.model.occ.addLine(10, 11, tag=8)
        self.model.occ.addLine(11, 12, tag=9)
        self.model.occ.addLine(13, 14, tag=10)
        self.model.occ.addLine(14, 15, tag=11)
        self.model.occ.addLine(15, 16, tag=12)
        self.model.occ.addLine(1, 5, tag=13)
        self.model.occ.addLine(5, 9, tag=14)
        self.model.occ.addLine(9, 13, tag=15)
        self.model.occ.addLine(2, 6, tag=16)
        self.model.occ.addLine(6, 10, tag=17)
        self.model.occ.addLine(10, 14, tag=18)
        self.model.occ.addLine(3, 7, tag=19)
        self.model.occ.addLine(7, 11, tag=20)
        self.model.occ.addLine(11, 15, tag=21)
        self.model.occ.addLine(4, 8, tag=22)
        self.model.occ.addLine(8, 12, tag=23)
        self.model.occ.addLine(12, 16, tag=24)

        if self.sa:
            self.model.occ.addLine(17, 1, tag=25)
            self.model.occ.addLine(18, 5, tag=26)
            self.model.occ.addLine(19, 9, tag=27)
            self.model.occ.addLine(20, 13, tag=28)
            self.model.occ.addLine(17, 18, tag=29)
            self.model.occ.addLine(18, 19, tag=30)
            self.model.occ.addLine(19, 20, tag=31)
            self.model.occ.addLine(4, 21, tag=32)
            self.model.occ.addLine(8, 22, tag=33)
            self.model.occ.addLine(12, 23, tag=34)
            self.model.occ.addLine(16, 24, tag=35)
            self.model.occ.addLine(21, 22, tag=36)
            self.model.occ.addLine(22, 23, tag=37)
            self.model.occ.addLine(23, 24, tag=38)

    def _create_faces(self) -> None:
        self.model.occ.addCurveLoop([1, 16, 4, 13], tag=1)
        self.model.occ.addSurfaceFilling(wireTag=1, tag=1, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([2, 19, 5, 16], tag=3)
        self.model.occ.addSurfaceFilling(wireTag=3, tag=2, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([3, 22, 6, 19], tag=5)
        self.model.occ.addSurfaceFilling(wireTag=5, tag=3, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([4, 17, 7, 14], tag=7)
        self.model.occ.addSurfaceFilling(wireTag=7, tag=4, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([5, 20, 8, 17], tag=9)
        self.model.occ.addSurfaceFilling(wireTag=9, tag=5, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([6, 23, 9, 20], tag=11)
        self.model.occ.addSurfaceFilling(wireTag=11, tag=6, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([7, 18, 10, 15], tag=13)
        self.model.occ.addSurfaceFilling(wireTag=13, tag=7, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([8, 21, 11, 18], tag=15)
        self.model.occ.addSurfaceFilling(wireTag=15, tag=8, degree=2, maxDegree=3)
        self.model.occ.addCurveLoop([9, 24, 12, 21], tag=17)
        self.model.occ.addSurfaceFilling(wireTag=17, tag=9, degree=2, maxDegree=3)

        if self.sa:
            self.model.occ.addCurveLoop([25, 13, 26, 29], tag=19)
            self.model.occ.addSurfaceFilling(wireTag=19, tag=10, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([26, 14, 27, 30], tag=21)
            self.model.occ.addSurfaceFilling(wireTag=21, tag=11, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([27, 15, 28, 31], tag=23)
            self.model.occ.addSurfaceFilling(wireTag=23, tag=12, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([32, 36, 33, 22], tag=25)
            self.model.occ.addSurfaceFilling(wireTag=25, tag=13, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([33, 37, 34, 23], tag=27)
            self.model.occ.addSurfaceFilling(wireTag=27, tag=14, degree=2, maxDegree=3)
            self.model.occ.addCurveLoop([34, 38, 35, 24], tag=29)
            self.model.occ.addSurfaceFilling(wireTag=29, tag=15, degree=2, maxDegree=3)


if __name__ == '__main__':
    # Define directories and paths
    work_dir = r"path/to/your/working/directory"
    geom_path = Path(work_dir) / 'uchannel4.step'

    # Instantiate the geometry
    u4 = UChannel4(geom_path, x_m=100, ll=25, lr=25, z_m=20, z_l=15, z_r=15, y_m=25, y_l=25, y_r=25, y_sa=12.5, r=0.0,
                   sa=True, alpha_l=0, alpha_r=0, beta_m=5, beta_l=5, beta_r=5)

    # Check the feasibility
    if u4.check_feasibility():
        # Create the geometry
        u4.gmsh_filleting = True
        u4.create_geometry(export=True, gui=True)
