from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import itertools
from PIL import Image
import gmsh

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    plot_radius = (limits[:, 1] - limits[:, 0]).max() / 2

    ax.set_xlim3d([centers[0] - plot_radius, centers[0] + plot_radius])
    ax.set_ylim3d([centers[1] - plot_radius, centers[1] + plot_radius])
    ax.set_zlim3d([centers[2] - plot_radius, centers[2] + plot_radius])

def create_imagegrid(data_dir: Path, outpath: Path) -> None:
    """
    Create an image grid from several single images.

    :param data_dir: Path to the directory with the parts (= part images)
    :param outpath: Path of the image grid file that should be created
    :return: big .png file with an approximately 16*9 ratio
    """
    # collect the images
    images_list = []
    for sub_dir in data_dir.iterdir():
        if sub_dir.is_dir():
            for img in sub_dir.iterdir():
                if img.is_file() and img.suffix == ".png":
                    images_list.append(img)

    # get image size
    img_w, img_h = Image.open(images_list[0]).size

    # rescale the images
    scale_factor = size_correlation(len(images_list))  # will scale down images depending on the image grid size
    with ProcessPoolExecutor() as executor:
        executor.map(resize_image, images_list, itertools.repeat(scale_factor))
        executor.shutdown()

    n_height = round(len(images_list) ** 0.5)
    n_width = n_height
    if not n_height * n_width >= len(images_list):
        n_width += 1

    # create imagegrid
    img_grid = Image.new('RGB', (n_width * img_w, n_height * img_h), color="white")
    index = 0
    for i in range(0, n_height * img_h, img_h):
        for j in range(0, n_width * img_w, img_w):
            im = Image.open(images_list[index])
            img_grid.paste(im, (j, i))
            index += 1

            if index == len(images_list):
                img_grid.save(outpath)
                break


def create_image(
    inpath: str,
    outpath: str,
    pov: str = "isometric x",
    title: str = "",
    zoom_factor: float = 1.0,
    show: bool = False,
):
    """
    Create an image of a geometry using gmsh. The background color is set to white.

    :param inpath: Path to the geometry
    :param outpath: Path to the exported image
    :param pov: Point of view (analogous to LS-PrePost)
    :param title: Text to print into the top of the image
    :param zoom_factor: >1.0 zooms in, <1.0 zooms out
    :param show: Option to show the image in the GUI
    """
    gmsh.initialize()
    gmsh.model.add("")
    gmsh.merge(inpath)

    white = (255, 255, 255)
    black = (0, 0, 0)
    gmsh.option.setColor("General.Background", *white)
    gmsh.option.setColor("General.Foreground", *black)
    gmsh.option.setColor("General.Text", *black)

    # Rotation
    pov_angles = {
        "front": (-90, 0, 90),
        "back": (-90, 0, 270),
        "left": (-90, 0, 0),
        "right": (-90, 0, 180),
        "top": (0, 0, 0),
        "bottom": (0, 0, 180),
        "isometric x": (-45, 0, 225),
        "isometric y": (45, 45, 90),
        "isometric z": (45, -45, 0),
        "isometric -x": (-45, 0, 45),
        "isometric -y": (45, 225, 90),
        "isometric -z": (45, 135, 0),
    }

    if pov not in pov_angles:
        raise ValueError(f"No such point of view: {pov}")
    x_rot, y_rot, z_rot = pov_angles[pov]

    gmsh.option.setNumber("General.Trackball", 0)
    gmsh.option.setNumber("General.RotationX", x_rot)
    gmsh.option.setNumber("General.RotationY", y_rot)
    gmsh.option.setNumber("General.RotationZ", z_rot)

    # Zoom
    for axis in ["ScaleX", "ScaleY", "ScaleZ"]:
        gmsh.option.setNumber(f"General.{axis}", zoom_factor)

    # Show or run in background
    if show:
        gmsh.fltk.run()
    else:
        gmsh.fltk.initialize()

    # Title annotation
    if title:
        gmsh.view.add("my_view", 1)
        gmsh.plugin.setString("Annotate", "Text", title)
        gmsh.plugin.setNumber("Annotate", "X", 1.e5)
        gmsh.plugin.setNumber("Annotate", "Y", 50)
        gmsh.plugin.setString("Annotate", "Font", "Times-BoldItalic")
        gmsh.plugin.setNumber("Annotate", "FontSize", 28)
        gmsh.plugin.setString("Annotate", "Align", "Center")
        gmsh.plugin.run("Annotate")

    gmsh.write(outpath)
    gmsh.fltk.finalize()


def resize_image(img_path: Path, factor: float):
    """
    Resize an image with constant aspect ratio by a factor.

    :param img_path: path to the .png image file
    :param factor: positive definite value ( >0 )
    :return: resized image
    """
    # sanity check
    if factor <= 0:
        raise ValueError("Only positive definite (greater than zero) allowed.")

    # open image
    im = Image.open(img_path)

    # resize image
    size = im.size
    new_size = (int(size[0] * factor), int(size[1] * factor))
    im_resized = im.resize(new_size)

    # export resized image
    im_resized.save(img_path.parent / f"{img_path.stem}_resized.png")


def size_correlation(n):
    """
    Small correlation for the calculation of the size factor of images, depending on how much images for an image grid
    are available.

    :param n: number of pictures in image grid
    :return: corrected value for image grid size
    """
    return 100 / (0.15 * n + 100)


if __name__ == "__main__":
    pass
