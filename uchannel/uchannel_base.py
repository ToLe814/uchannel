from pathlib import Path
import sys
import math
from typing import List
from collections import namedtuple

import gmsh
import yaml
import numpy as np

from uchannel.error_handling import FeasibilityError


class UChannelBase:
    """
    Base class for all UChannel classes. This class provides the basic structure.
    """
    def __init__(self, path: Path, **kwargs):
        self.kwargs = kwargs
        self.path = path
        self.name = path.stem
        self.model = None
        self.gmsh_filleting: bool = True   # Flag to use gmsh radius

        # Define named tuples for the vertices
        self.Vertex = namedtuple('Vertex', ['x', 'y', 'z'])
        self.Bounds = namedtuple("bounds", ["min", "max"])

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
    def _validate_parameter(parameter_name: str, value: float, min_value: float, max_value: float) -> None:
        """
        Validate the given parameter value. If the value is not within the given range, raise a FeasibilityError.
        :param parameter_name: name of the parameter
        :param value: value of the parameter
        :param min_value: minimum value of the parameter
        :param max_value: maximum value of the parameter
        :return: None
        """
        if not (min_value <= value <= max_value):
            if "alpha" in parameter_name or "beta" in parameter_name:  # angles
                raise FeasibilityError(f'{parameter_name} must be between {math.degrees(min_value)} and'
                                       f' {math.degrees(max_value)}.')
            elif parameter_name in ["r_r", "r_l", "r"] and value in [0, 0.0]:  # radii
                # omit bounds check for sharp edges
                pass
            else:
                raise FeasibilityError(f'{parameter_name} must be between {min_value} and {max_value}.')

    def check_feasibility(self):
        raise NotImplementedError

    def _export_geometry(self) -> None:
        gmsh.write(str(self.path))

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

    @staticmethod
    def create_image(inpath: Path, outpath: Path, pov="isometric x", title="", zoom_factor=1.0, show=False):
        """
        Create an image of a geometry using gmsh. The background color is set to white.

        :param inpath: path to the geometry
        :param outpath: path to the exported image
        :param pov: point of view (analogous to LS-PrePost)
        :param title: Text to print into the top of the image
        :param zoom_factor: zoom_factor > 1.0: zoom in
                            zoom_factor < 1.0: zoom out
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
        gmsh.option.setNumber("General.ScaleX", zoom_factor)
        gmsh.option.setNumber("General.ScaleX", zoom_factor)

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
