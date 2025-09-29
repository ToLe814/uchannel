from pathlib import Path
import logging
import math

import numpy as np
from pyDOE3 import lhs
from joblib import Parallel, delayed
import gmsh

from uchannel.uchannel1 import UChannel1
from uchannel.uchannel2 import UChannel2
from uchannel.uchannel3 import UChannel3
from uchannel.uchannel4 import UChannel4


class Sampler:
    """ Class to generate samples for a given UChannel class using Latin Hypercube Sampling. """

    def __init__(self, uchannel_class, fillet_method: str, n_samples: int, data_dir: Path, limiting_parameter: float):
        """
        Initialize the sampler class.

        :param uchannel_class: UChannel class to generate samples for
        :param n_samples: number of samples to generate
        :param data_dir: directory to store the samples
        :param limiting_parameter: value of the limiting parameter (UChannel1: x_1, UChannel2: x_m, UChannel3: x_b1,
                                                                    UChannel4: x_m)
        """
        self.uchannel = uchannel_class
        self.fillet_method = fillet_method
        self.n_samples = n_samples
        self.data_dir = data_dir
        self.limiting_parameter = limiting_parameter
        self.bounds = {}

    def get_bounds(self) -> None:
        """ Get the parameter bounds for Latin Hypercube Sampling (LHS) for the given UChannel class."""
        self.bounds = self.uchannel(Path(), Path(), self.limiting_parameter).f_bounds

    def _get_params(self, sample: list) -> dict:
        """
        Get the parameters for the given sample.

        :param sample: sample index
        :return: dictionary of parameter names and values
        """
        if self.uchannel == UChannel1:
            params = {
                "x_1": self.limiting_parameter,
                "x_2": sample[0],
                "y_l": sample[1],
                "y_r": sample[2],
                "y_fl": sample[3],
                "z_l_1": sample[4],
                "z_l_2": sample[5],
                "z_r_1": sample[6],
                "z_r_2": sample[7],
                "o_mp_1": (sample[8], sample[9], sample[10]),
                "o_mp_2": (sample[11], sample[12], sample[13]),
                "r_l": (sample[14], sample[15]),
                "r_r": (sample[16], sample[17]),
                # convert radians → degrees for angles
                "beta_l_1": math.degrees(sample[18]),
                "beta_l_2": math.degrees(sample[19]),
                "beta_r_1": math.degrees(sample[20]),
                "beta_r_2": math.degrees(sample[21]),
                # handle flange as boolean (rounding ensures clean conversion)
                "fl": bool(round(sample[22]))
            }
        elif self.uchannel == UChannel2:
            params = {
                "x_m": self.limiting_parameter,
                "x_l": sample[0],
                "x_r": sample[1],
                "ll": sample[2],
                "lr": sample[3],
                "z_m": sample[4],
                "z_l": sample[5],
                "z_r": sample[6],
                "y_m": sample[7],
                "y_l": sample[8],
                "y_r": sample[9],
                "y_fl": sample[10],
                "alpha_l": math.degrees(sample[11]),
                "alpha_r": math.degrees(sample[12]),
                "beta_l": math.degrees(sample[13]),
                "beta_r": math.degrees(sample[14]),
                "beta_m": math.degrees(sample[15]),
                "r": sample[16],
                "fl": bool(round(sample[17])),
                "la": bool(round(sample[18])),
                "ra": bool(round(sample[19]))
            }
        elif self.uchannel == UChannel3:
            params = {
                "x_b1": self.limiting_parameter,
                "x_b2": sample[0],
                "x_ad": sample[1],
                "l_c1": sample[2],
                "l_c2": sample[3],
                "x_t": sample[4],
                "y_m": sample[5],
                "alpha": math.degrees(sample[6]),
                "z_ad": sample[7],
                "z_b": sample[8],
                "z_t": sample[9],
                "beta_b": math.degrees(sample[10]),
                "beta_c": math.degrees(sample[11]),
                "beta_t": math.degrees(sample[12]),
                "y_fl": sample[13],
                "r": sample[14],
                "ad": bool(round(sample[15])),
                "fl": bool(round(sample[16]))
            }
        elif self.uchannel == UChannel4:
            params = {
                "x_m": self.limiting_parameter,
                "ll": sample[0],
                "lr": sample[1],
                "z_m": sample[2],
                "z_l": sample[3],
                "z_r": sample[4],
                "y_m": sample[5],
                "y_l": sample[6],
                "y_r": sample[7],
                "y_fl": sample[8],
                "r": sample[9],
                "fl": bool(round(sample[10])),
                "alpha_l": math.degrees(sample[11]),
                "alpha_r": math.degrees(sample[12]),
                "beta_m": math.degrees(sample[13]),
                "beta_l": math.degrees(sample[14]),
                "beta_r": math.degrees(sample[15])
            }
        else:
            raise NotImplementedError("The given UChannel class is not supported.")

        return params

    def generate_samples(self, gen_image: bool = True, parallel: bool = False) -> None:
        """
        Generate samples using LHS.

        :param gen_image: Flag to generate images
        :param parallel: Flag to generate the samples in parallel
        """
        if self.bounds is None:
            self.get_bounds()

        # get factors of optimized Latin Hypercube Sampling
        n_parameters = len(self.bounds)
        lhd = lhs(n=n_parameters, samples=self.n_samples, criterion="maximin", iterations=100)

        # merge factors to bounds
        # bounds = np.array(list(self.bounds.values()))
        bounds = np.array([(b.min, b.max) for b in self.bounds.values()])
        samples = bounds[:, 0] + lhd * (bounds[:, 1] - bounds[:, 0])  # lb + factor * (ub - lb)

        # process sample (parallel or serial)
        if parallel:
            Parallel(n_jobs=-1)(delayed(self._process_sample)(sample, i, gen_image) for i, sample in enumerate(samples))
        else:
            for i, sample in enumerate(samples):
                try:
                    self._process_sample(sample, i, gen_image)
                except:  # any gmsh error is excepted
                    gmsh.finalize()

    def _process_sample(self, sample: list, i: int, gen_image: bool) -> None:
        """
        Process a single sample.

        :param sample: Sample values
        :param i: Sample index
        :param gen_image: Flag to generate images
        """
        sample_folder = self.data_dir / f"sample_{i}"
        geom_path = sample_folder / f"sample_{i}.step"
        label_path = sample_folder / f"sample_{i}.json"
        image_path = sample_folder / f"sample_{i}.png"
        yaml_path = sample_folder / "parameters.yaml"

        params = self._get_params(sample)
        u = self.uchannel(geom_path, label_path, **params)
        u.fillet_method = self.fillet_method
        if u.check_feasibility():
            sample_folder.mkdir(exist_ok=True, parents=True)
            u.export_parameter(params=params, output_path=yaml_path)
            u.create_geometry(export=True, gui=False, segmentation_labels=False)
            if gen_image:
                u.create_image(input_path=geom_path, output_path=image_path,
                               pov="isometric x", title=f"Sample {i}", zoom_factor=0.8, show=False)
        else:
            logging.warning(params)
            logging.warning(f"Sample {i} is not feasible.")


if __name__ == '__main__':
    from functions import create_imagegrid
    import os
    logging.basicConfig(level=logging.INFO)

    # # Generate samples of UChannel1
    # set directories
    work_dir = r"path/to/your/working/directory"
    data_dir = Path(work_dir) / 'uchannel1_data'
    os.makedirs(data_dir, exist_ok=True)

    # instantiate Sampler
    u1_sampler = Sampler(UChannel1, fillet_method="explicit", n_samples=250, data_dir=data_dir, limiting_parameter=400)

    # get the bounds for the remaining parameters for the given limiting parameter
    u1_sampler.get_bounds()

    # generate the samples
    u1_sampler.generate_samples(gen_image=True, parallel=True)

    # Create an Imagegrid of the generated samples
    path_imagegrid = Path(data_dir) / 'imagegrid.png'
    create_imagegrid(data_dir=data_dir, outpath=path_imagegrid)
