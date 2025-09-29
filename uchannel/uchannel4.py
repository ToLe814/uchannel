from pathlib import Path
import math
import gmsh
import networkx as nx

from uchannel.uchannel_base import UChannelBase
from uchannel.error_handling import FeasibilityError


class UChannel4(UChannelBase):
    def __init__(self, path: Path, label_path: Path, x_m: float = 100, ll: float = 50, lr: float = 50, z_m: float = 40, z_l: float = 53.3,
                 z_r: float = 53.3, y_m: float = 50, y_l: float = 62.5, y_r: float = 62.5, y_fl: float = 20,
                 r: float = 12.5, fl: bool = True, alpha_l: float = 25, alpha_r: float = 25, beta_m: float = 25,
                 beta_l: float = 15, beta_r: float = 15):
        """
        Class for generating the UChannel4 geometry in gmsh. A short description of the parameters is given below.

        The key parameter for feasibility control is x_m. It can take any value. The authors recommend values between
        189 and 1223 mm. The remaining parameters are limited by x_m.

        :param path: Path to the .step file where the geometry will be exported
        :param label_path: Path to the .json file where the segmentation labels will be exported
        :param x_m: Length of the middle plane m in x direction
        :param ll: Length of the left plane l in normal direction
        :param lr: Length of the right plane r in normal direction
        :param z_m: Height of the middle plane m in z direction
        :param z_l: Height of the left plane l in z direction
        :param z_r: Height of the right plane r in z direction
        :param y_m: Width of the middle plane m in y direction
        :param y_l: Width of the left plane l in y direction
        :param y_r: Width of the right plane r in y direction
        :param y_fl: Width of the flange in y direction
        :param r: Radius of the fillets. Same for all fillets.
        :param fl: Boolean to decide if the flange should be added
        :param alpha_l: Angle of the left plane l from the x-axis in degrees
        :param alpha_r: Angle of the right plane r from the x-axis in degrees
        :param beta_m: Angle of the slant (middle plane) in degrees
        :param beta_l: Angle of the slant (left plane) in degrees
        :param beta_r: Angle of the slant (right plane) in degrees
        """
        super().__init__(path, label_path)

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
        self.y_fl = y_fl
        self.r = r
        if isinstance(fl, float):
            self.fl = bool(round(fl))
        elif isinstance(fl, bool):
            self.fl = fl
        self.alpha_l = math.radians(alpha_l)
        self.alpha_r = math.radians(alpha_r)
        self.beta_m = math.radians(beta_m)
        self.beta_l = math.radians(beta_l)
        self.beta_r = math.radians(beta_r)

        self._calculate_vertices()
        self.f_bounds = self._feasibility_bounds()  # Feasibility bounds

    def _calculate_vertices(self) -> None:
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

        # Vertices that construct the left flange
        self.v17 = self.Vertex(self.v1.x, self.v1.y + self.y_fl, self.v1.z)
        self.v18 = self.Vertex(self.v5.x, self.v5.y + self.y_fl, self.v5.z)
        self.v19 = self.Vertex(self.v9.x, self.v9.y + self.y_fl, self.v9.z)
        self.v20 = self.Vertex(self.v13.x, self.v13.y + self.y_fl, self.v13.z)

        # Vertices that construct the right slant
        self.v4 = self.Vertex(self.v3.x, self.v3.y - y_s_l, self.v3.z + self.z_l)
        self.v8 = self.Vertex(self.v7.x, self.v7.y - y_s, self.v7.z + self.z_m)
        self.v12 = self.Vertex(self.v11.x, self.v11.y - y_s, self.v11.z + self.z_m)
        self.v16 = self.Vertex(self.v15.x, self.v15.y - y_s_r, self.v15.z + self.z_r)

        # Vertices that construct the right flange
        self.v21 = self.Vertex(self.v4.x, self.v4.y - self.y_fl, self.v4.z)
        self.v22 = self.Vertex(self.v8.x, self.v8.y - self.y_fl, self.v8.z)
        self.v23 = self.Vertex(self.v12.x, self.v12.y - self.y_fl, self.v12.z)
        self.v24 = self.Vertex(self.v16.x, self.v16.y - self.y_fl, self.v16.z)

    def _feasibility_bounds(self) -> dict:
        """
        Bounds for the feasibility check. Can be used as an input for an LHS sampler.
        """
        bounds = {
            "ll": self.Bounds(1/4*self.x_m, 1/2*self.x_m),
            "lr": self.Bounds(1/4*self.x_m, 1/2*self.x_m),
            "z_m": self.Bounds(1/5 * self.x_m, 3/5 * self.x_m),
            "z_l": self.Bounds(1/5 * self.x_m, 3/5 * self.x_m),
            "z_r": self.Bounds(1/5 * self.x_m, 3/5 * self.x_m),
            "y_m": self.Bounds(1/4*self.x_m, 1/2*self.x_m),
            "y_l": self.Bounds(1/4*self.x_m, 3/8*self.x_m),
            "y_r": self.Bounds(1/4*self.x_m, 3/8*self.x_m),
            "y_fl": self.Bounds(1/8*self.x_m, 1/5*self.x_m),
            "r": self.Bounds(1/150 * self.x_m, 0.08 * self.x_m),
            "fl": self.Bounds(True, False),
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
            self._validate_parameter('y_fl', self.y_fl, self.f_bounds['y_fl'].min, self.f_bounds['y_fl'].max)
            self._validate_parameter('alpha_l', self.alpha_l, self.f_bounds['alpha_l'].min, self.f_bounds['alpha_l'].max)
            self._validate_parameter('alpha_r', self.alpha_r, self.f_bounds['alpha_r'].min, self.f_bounds['alpha_r'].max)

            # Checks for manufacturing constraints
            self._validate_parameter('beta_m', self.beta_m, self.f_bounds['beta_m'].min, self.f_bounds['beta_m'].max)
            self._validate_parameter('beta_l', self.beta_l, self.f_bounds['beta_l'].min, self.f_bounds['beta_l'].max)
            self._validate_parameter('beta_r', self.beta_r, self.f_bounds['beta_r'].min, self.f_bounds['beta_r'].max)

        except FeasibilityError as e:
            print(f"FeasibilityError: {e}")
            return False

        return True

    def create_geometry(self, segmentation_labels: bool = True, export: bool = True, gui: bool = False) -> None:
        """
        Create the UChannel4 geometry in gmsh using the parameters given in the constructor of UChannel4 and the
        calculated vertices.

        :param segmentation_labels: Extract the segmentation labels from the geometry.
        :param export: Export the geometry to the path given in the constructor of UChannel4
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
            # Edges that need fillets weather fl is true or false.
            fillet_edges_tags = [4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21]

            if self.fl:
                fillet_edges_tags.extend([26, 27, 33, 34, 13, 14, 15, 22, 23, 24])

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
        """ Create all vertices of the geometry """
        # Make sure we have a graph to work with
        if not hasattr(self, "graph"):
            self.graph = nx.Graph()

        # Collect all vertex attribute names
        vertex_names = [f"v{i}" for i in range(1, 17)]
        if self.fl:
            vertex_names.extend([f"v{i}" for i in range(17, 25)])

        # Flange radii tags. Which are needed for explicit filleting
        self.flange_radii_tags = [1, 5, 9, 13, 16, 12, 8, 4]

        for idx, name in enumerate(vertex_names, start=1):
            v = getattr(self, name)  # e.g. self.v1, self.v2 ...
            tag = self.model.occ.addPoint(v.x, v.y, v.z, tag=idx)

            # Add a node to the graph with gmsh tag + coords
            self.graph.add_node(
                name,
                tag=tag,
                coords=(v.x, v.y, v.z),
            )

    def _create_edges(self) -> None:
        """ Create all edges of the geometry. """
        if not hasattr(self, "graph"):
            raise RuntimeError("Graph not initialized. Call _create_vertices first.")

        # Base edges
        edge_specs = [
            ("e1", "v1", "v2"),
            ("e2", "v2", "v3"),
            ("e3", "v3", "v4"),
            ("e4", "v5", "v6"),
            ("e5", "v6", "v7"),
            ("e6", "v7", "v8"),
            ("e7", "v9", "v10"),
            ("e8", "v10", "v11"),
            ("e9", "v11", "v12"),
            ("e10", "v13", "v14"),
            ("e11", "v14", "v15"),
            ("e12", "v15", "v16"),
            ("e13", "v1", "v5"),
            ("e14", "v5", "v9"),
            ("e15", "v9", "v13"),
            ("e16", "v2", "v6"),
            ("e17", "v6", "v10"),
            ("e18", "v10", "v14"),
            ("e19", "v3", "v7"),
            ("e20", "v7", "v11"),
            ("e21", "v11", "v15"),
            ("e22", "v4", "v8"),
            ("e23", "v8", "v12"),
            ("e24", "v12", "v16"),
        ]

        # Flange edges
        if self.fl:
            edge_specs += [
                ("e25", "v17", "v1"),
                ("e26", "v18", "v5"),
                ("e27", "v19", "v9"),
                ("e28", "v20", "v13"),
                ("e29", "v17", "v18"),
                ("e30", "v18", "v19"),
                ("e31", "v19", "v20"),
                ("e32", "v4", "v21"),
                ("e33", "v8", "v22"),
                ("e34", "v12", "v23"),
                ("e35", "v16", "v24"),
                ("e36", "v21", "v22"),
                ("e37", "v22", "v23"),
                ("e38", "v23", "v24"),
            ]

        # Create edges in gmsh and add them to the graph
        for idx, (ename, n1, n2) in enumerate(edge_specs, start=1):
            tag = self.model.occ.addLine(
                self.graph.nodes[n1]["tag"],
                self.graph.nodes[n2]["tag"],
                tag=idx
            )
            self.graph.add_edge(
                n1, n2,
                name=ename,
                tag=tag,
                kind="line"
            )

        # For the explicit fillet method, mark the helper edges as those will be specifically removed later
        if self.fillet_method == "explicit":
            if self.fl:
                helper_tags = (4, 26, 5, 6, 33, 27, 7, 8, 9, 34, 19, 20, 21, 16, 17, 18, 13, 14, 15, 22, 23, 24)
            else:
                helper_tags = (4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21)
            for u, v, d in self.graph.edges(data=True):
                if d.get("tag") in helper_tags:
                    d["is_helper"] = True

    def _create_surfaces_procedural(self) -> None:
        """ Create all surfaces of the geometry using procedural construction """
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

        if self.fl:
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
        self._connect_chain("bottom")
        self._connect_chain("top")

        # special needed in the case of alpha values ~0
        removed_base_node_tags = []
        for u, v, data in self.graph.edges(data=True):
            if "arc" and "collinear" in data["name"]:
                adjacencies_u, _  = self.model.getAdjacencies(0, self.graph.nodes[u]["tag"])
                adjacencies_v, _ = self.model.getAdjacencies(0, self.graph.nodes[v]["tag"])
                if len(adjacencies_u) < 2 and len(adjacencies_v) < 2:
                    removed_base_node_tags.append(self.graph.nodes[u]["base"])
                    removed_base_node_tags.append(self.graph.nodes[v]["base"])
                    self.model.occ.remove([(1, data["tag"])], recursive=True)

        self.model.occ.synchronize()

        self._remove_helper_edges()
        self._fragment_at_graph_nodes()

        self._rebuild_graph_from_occ()

        self._clean_graph()
        surfaces = self._build_surfaces_from_ordered_cycles()
        self._build_shell_from_surfaces(surfaces)


if __name__ == '__main__':
    # Define directories and paths
    work_dir = r"path/to/your/working/directory"
    geom_path = Path(work_dir) / 'uchannel4.step'
    label_path = Path(work_dir) / 'uchannel4.json'

    # Instantiate the geometry
    u4 = UChannel4(geom_path, label_path, x_m=100, ll=25, lr=25, z_m=20, z_l=15, z_r=15, y_m=25, y_l=25, y_r=25,
                   y_fl=12.5, r=0.0, fl=True, alpha_l=0, alpha_r=0, beta_m=5, beta_l=5, beta_r=5)

    # Check the feasibility
    if u4.check_feasibility():
        # Create the geometry
        u4.fillet_method = 'explicit'
        u4.create_geometry(segmentation_labels=True, export=True, gui=True)
        u4.visualize_segmentation_graph()