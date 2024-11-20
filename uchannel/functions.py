from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools

from PIL import Image
import gmsh


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


def create_image(inpath: str, outpath: str, pov="isometric x", title="", zoom_factor=1.0, show=False):
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
    gmsh.merge(inpath)

    # set background color
    white = (255, 255, 255)
    black = (0, 0, 0)
    gmsh.option.setColor("General.Background", white[0], white[1], white[2])
    gmsh.option.setColor("General.Foreground", black[0], black[1], black[2])
    gmsh.option.setColor("General.Text", black[0], black[1], black[2])

    # rotate to wanted perspective
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
        raise ValueError(f'There exists no point of view: {pov}. Please adapt it.')

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
    gmsh.write(outpath)
    gmsh.fltk.finalize()
    gmsh.finalize()


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
