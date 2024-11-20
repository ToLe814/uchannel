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

    def __init__(self, uchannel_class, gmsh_filleting: bool, n_samples: int, data_dir: Path, limiting_parameter: float):
        """
        Initialize the sampler class.

        :param uchannel_class: UChannel class to generate samples for
        :param n_samples: number of samples to generate
        :param data_dir: directory to store the samples
        :param limiting_parameter: value of the limiting parameter (UChannel1: x_1, UChannel2: x_m, UChannel3: x_b1,
                                                                    UChannel4: x_m)
        """
        self.uchannel = uchannel_class
        self.gmsh_filleting = gmsh_filleting
        self.n_samples = n_samples
        self.data_dir = data_dir
        self.limiting_parameter = limiting_parameter
        self.bounds = {}

    def get_bounds(self) -> None:
        """ Get the parameter bounds for Latin Hypercube Sampling (LHS) for the given UChannel class."""
        self.bounds = self.uchannel(Path(), self.limiting_parameter).f_bounds

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
                "y_sa": sample[3],
                "z_l_1": sample[4],
                "z_l_2": sample[5],
                "z_r_1": sample[6],
                "z_r_2": sample[7],
                "offset_mp_1": (sample[8], sample[9], sample[10]),
                "offset_mp_2": (sample[11], sample[12], sample[13]),
                "r_l": (sample[14], sample[15]),
                "r_r": (sample[16], sample[17]),
                "beta_l_1": math.degrees(sample[18]),
                "beta_l_2": math.degrees(sample[19]),
                "beta_r_1": math.degrees(sample[20]),
                "beta_r_2": math.degrees(sample[21]),
                "sa": sample[22]
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
                "y_sa": sample[10],
                "alpha_l": math.degrees(sample[11]),
                "alpha_r": math.degrees(sample[12]),
                "beta_l": math.degrees(sample[13]),
                "beta_r": math.degrees(sample[14]),
                "beta_m": math.degrees(sample[15]),
                "r": sample[16],
                "sa": sample[17],
                "la": sample[18],
                "ra": sample[19]
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
                "y_sa": sample[13],
                "r": sample[14],
                "ad": sample[15],
                "sa": sample[16]
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
                "y_sa": sample[8],
                "r": sample[9],
                "sa": sample[10],
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
        bounds = np.array(list(self.bounds.values()))
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
        image_path = sample_folder / f"sample_{i}.png"
        yaml_path = sample_folder / "parameters.yaml"

        params = self._get_params(sample)
        u = self.uchannel(geom_path, **params)
        u.gmsh_filleting = self.gmsh_filleting

        if u.check_feasibility:
            sample_folder.mkdir(exist_ok=True, parents=True)
            u.export_parameter(params=params, outpath=yaml_path)
            u.create_geometry(export=True, gui=False)
            if gen_image:
                u.create_image(inpath=geom_path, outpath=image_path,
                               pov="isometric x", title=f"Sample {i}", zoom_factor=1.0, show=False)
        else:
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
    u1_sampler = Sampler(UChannel1, gmsh_filleting=True, n_samples=250, data_dir=data_dir, limiting_parameter=400)

    # get the bounds for the remaining parameters for the given limiting parameter
    u1_sampler.get_bounds()

    # generate the samples
    u1_sampler.generate_samples(gen_image=True, parallel=True)

    # Create an Imagegrid of the generated samples
    path_imagegrid = Path(data_dir) / 'imagegrid.png'
    create_imagegrid(data_dir=data_dir, outpath=path_imagegrid)
