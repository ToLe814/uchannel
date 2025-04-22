from pathlib import Path
import sys
import math
from typing import List, Optional, Union, Tuple
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gmsh
import yaml
import numpy as np
import json

from uchannel.error_handling import FeasibilityError
from uchannel.segmentation import FaceAdjacencyGraph, FaceSegmenter
from uchannel.functions import set_axes_equal


class UChannelBase:
    """
    Base class for all UChannel classes. This class provides the basic structure.
    """
    def __init__(self, geom_path: Path, label_path: Path, **kwargs):
        self.kwargs = kwargs
        self.geom_path = geom_path
        self.label_path = label_path
        self.name = geom_path.stem
        self.model = None
        self.gmsh_filleting: bool = True   # Flag to use gmsh radius
        self.face_adjacency = FaceAdjacencyGraph()
        self.graph = None
        self.face_segmenter = None
        self.segmentation_labels: dict = {}

        # Define named tuples for the vertices
        self.Vertex = namedtuple('Vertex', ['x', 'y', 'z'])
        self.Bounds = namedtuple("bounds", ["min", "max"])
        self.surface_labels = {}

    @staticmethod
    def create_fillet(edges: List[int], radii: List[float]) -> None:
        """
        Create a fillet on the given edges with the given radius. The
        radii vector can either contain a single radius, as many radii as edges, or
        twice as many as edges (in which case different radii are provided for the start
        and end points of the curves)
        :param edges: curveTags of the edges to fillet
        :param radii: radii of the fillets
        :return: model with fillets
        """
        # generate a volume from the surfaces
        sf_tags = [tag for dim, tag in gmsh.model.occ.getEntities(2)]
        sl_tag = gmsh.model.occ.addSurfaceLoop(sf_tags, 1)
        v_tag = gmsh.model.occ.addVolume([sl_tag], 1)

        # create the fillet and remove the volume
        if len(radii) == 1 and radii[0] in [0, 0.0]:  # if there is a single radius of 0 for every edge, do not fillet
            pass
        elif sum(radii) == 0:  # if there are no radii, do not fillet
            pass
        else:
            try:
                gmsh.model.occ.fillet([v_tag], edges, radii, removeVolume=True)
            except Exception as e:
                print(f'Error while creating fillet: {e}')
                return

    @staticmethod
    def _validate_parameter(parameter_name: Union[str, Tuple[str, str]], value: Union[float, Tuple[float, float]], min_value: float, max_value: float) -> None:
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
                # omit bounds check for sharp edges
                pass
            else:
                raise FeasibilityError(f'{parameter_name} must be between {min_value} and {max_value}.')

    def check_feasibility(self):
        raise NotImplementedError

    def _export_geometry(self) -> None:
        gmsh.write(str(self.geom_path))

    def _write_segmentation_labels(self) -> None:
        """
        Export the segmentation labels to a JSON file using the gmsh face idx

        :return: None
        """
        with open(self.label_path, 'w') as f:
            json.dump(self.segmentation_labels, f, indent=4)

    @staticmethod
    def export_parameter(params: dict, outpath: Path) -> None:
        """
        Save parameters as YAML.
        :param params: Dictionary containing parameter names and values
        :param outpath: File path to save the YAML file
        """
        # Convert boolean values to strings, handling both Python bool and numpy.bool_
        for key, value in params.items():
            if isinstance(value, (bool, np.bool_)):
                params[key] = "True" if value else "False"

            if isinstance(value, tuple):
                params[key] = list(value)

        with open(outpath, 'w') as yaml_file:
            yaml.dump(params, yaml_file, indent=4)

    @staticmethod
    def export_fillet_edges(edges: List, outpath: Path) -> None:
        """
        Export the fillet vertices list to a txt file.
        :param edges: List of edges
        :param outpath: File path to save the txt file
        """
        # if outpath is not a txt file, raise an error
        if outpath.suffix != '.txt':
            raise ValueError(f'Output path must be a txt file. Got: {outpath}')

        with open(outpath, 'w') as file:
            file.write(str(edges))

    @staticmethod
    def _visualize_geometry() -> None:
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()

    def create_geometry(self):
        raise NotImplementedError

    def get_segmentation_labels(self, has_fillet: Optional[bool] = None, has_flange: Optional[bool] = None) -> dict:
        """
        Compute segmentation labels using the FaceSegmenter.

        :param has_fillet: Optional override for fillet detection.
        :param has_flange: Optional override for flange detection.
        :return: Dict of classified face labels.
        """
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
            has_flange = getattr(self, "sa", False)

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
    def create_image(inpath: Path, outpath: Path, pov="isometric x", title="", zoom_factor=1.0, curve_width=2.0,
                     show=False):
        """
        Create an image of a geometry using gmsh. The background color is set to white.

        :param inpath: path to the geometry
        :param outpath: path to the exported image
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
        gmsh.merge(str(inpath))

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
        gmsh.write(str(outpath))
        gmsh.fltk.finalize()
        gmsh.finalize()


if __name__ == '__main__':
    pass
