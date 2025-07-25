# U-Channel

![Status](https://img.shields.io/badge/status-stable-green) ![Version](https://img.shields.io/badge/version-1.2.0-blue) ![License](https://img.shields.io/github/license/ToLe814/uchannel?cacheSeconds=43200) ![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=ToLe814.uchannel) ![GitHub Repo stars](https://img.shields.io/github/stars/ToLe814/uchannel?color=blue&style=flat-square&cacheSeconds=30)


Welcome to the **U-Channel** project, an [installable Python package](#installation) designed to provide parametric CAD models for the generation of U-shaped sheet metal geometries.

<p align="center">
<img src="/uchannel.gif" width="550"/>
</p>

---

## Table of Contents

1. [Overview](#overview)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Contributing](#contributing)  
5. [Cite This Work](#cite-this-work)  
6. [License](#license)  

---

## Overview

The U-Channel Software is a Python package designed for generating and working with parametric models of U-shaped sheet metals. The package includes four distinct parametric models, each defined by geometric dimensions, angles, and fillet radii. These models enable users to create an infinite variety of U-shaped sheet metal geometries, which can be exported as STEP files for CAE applications.
The key features of the package are:

- Parametric Models: Generate custom U-shaped geometries inspired by real-world shapes.
- Feasibility Check: Includes a plausibility check to validate the geometries.
- Segmentation: Automatically segment faces into bottom, slant, fillet, and flange regions
- Visualization: Create detailed images of the generated geometries.
- Parallel Processing: Efficiently generate geometries and images in parallel.

For more information, please refer to the associated [publication(s)](#cite-this-work). This project is currently classified as **stable**, which means that it can be used as a [package](#installation) in its current state. Changes, updates, and new features may occur over time in the form of new versions.

---


## Installation

To install the package, you can use the following command via pip to install the latest version:

```bash
pip install git+https://github.com/ToLe814/uchannel.git -U
```

If you want to install a specific version, adapt this pip installation command:
```bash
pip install git+https://github.com/ToLe814/uchannel.git@1.0.0
```

Alternatively, you can clone the repository and install the package locally:
```bash
git clone https://github.com/ToLe814/uchannel.git
cd uchannel
pip install .
```


## Requirements
We refer to the pyproject.toml file.	
	

## Usage
Here’s a quick example of using the U-Channel package with one of the four parametric models after installation:
```python
from uchannel.uchannel1 import UChannel1
from pathlib import Path

# Define directories and paths
work_dir = r"path/to/your/working/directory"
path_geom = Path(work_dir) / 'uchannel1.step'
label_path = Path(work_dir) / 'uchannel1.json'

# Instantiate the geometry
u1 = UChannel1(path_geom, label_path, x_1=100, x_2=105, y_l=50, y_r=50, z_l_1=33.3, z_l_2=33.3, z_r_1=33.3, z_r_2=33.3,
               o_mp_1=(1, 1, 1), o_mp_2=(1, 1, 1), r_l=(5, 5), r_r=(5, 5),
               beta_l_1=15, beta_l_2=15, beta_r_1=15, beta_r_2=15, y_sa=50, sa=True)

# Check the feasibility
if u1.check_feasibility():
        # Create the geometry
        u1.gmsh_filleting = True
        u1.create_geometry(segmentation_labels=True, export=True, gui=False)
```
Find running examples of feasible geometry creation in the respective project files. Explanation for the parameters can be found in the comments, [sketches](./sketches/), or in the associated [publication(s)](#cite-this-work). Currently, there is no documentation of the code.


## Contributing

We appreciate your interest in contributing to the U-Channel project! Before submitting a pull request, please ensure that:

    Your code adheres to the coding style used in the repository.
    Any new features or changes align with the project's goals.

Pull requests are considered within the frame of the above-described requirements. Please note that this project does not receive any funding. As a result, responses and updates may take some time. Thank you for your understanding.


## Theoretical Background

For motivation and theoretical background on the model definitions, see:

```bib
@article{Lehrer.2025,
 author = {Lehrer, Tobias and Stocker, Philipp and Duddeck, Fabian and Wagner, Marcus},
 year = {2025},
 title = {UCSM: Dataset of U-shaped parametric CAD geometries and real-world sheet metal meshes for deep drawing},
 pages = {103924},
 issn = {00104485},
 journal = {Computer-Aided Design},
 doi = {10.1016/j.cad.2025.103924}
}
```

## Cite This Work

If you use this software in your research or applications, please cite it as follows:

```bib
@misc{Lehrer.2024,
 abstract = {The U-Channel Software is a python package containing four parametric models of U-shaped Sheet metals. The parameters refer to geometric dimensions, angles, fillet radii. Using the models, one can generate an infinite number of U-shaped sheet metal geometries, e.g. as STEP files. The parametric models are inspired by real-world shapes. A feasibility check is provided to ensure plausiblity of the shapes. Further features of the code are creating images of the geometries and generating geometries and images in parallel. For more details, refer to the repository and associated publications.},
 author = {Lehrer, Tobias and Stocker, Philipp and Duddeck, Fabian and Wagner, Marcus},
 year = {2024},
 title = {Parametric CAD Models to Create an Infinite Number of Structural U-Shaped Sheet Metal Geometries},
 publisher = {Zenodo},
 doi = {10.5281/zenodo.14191374}
}
```

Moreover, a ready-to-use dataset of 2533 parts with U-Channel-1 to U-Channel-4 geometries is available here:

```bib
@misc{Lehrer.2025b,
 abstract = {The dataset contains 2533 geometries from four different parametric CAD models sampled from the U-Channel python package. The representations include the original CAD geometries (.step), graph binaries (.bin), meshes (.off), and point clouds (.xyz). Additionally, we provide labels for supervised learning use cases of local and global drawability assessment and part segmentation (.json) for all representations. Custom label computation for drawability assessment is enabled by the provided strains.zip file, which contains minor and major true strains for the given meshes. For more details, refer to the associated publication.},
 author = {Lehrer, Tobias and Stocker, Philipp and Duddeck, Fabian and Wagner, Marcus},
 date = {2025},
 title = {Machine Learning Dataset of U-Channel Sheet Metal Geometry Representations with Supervision Information for Drawability Assessment and Part Segmentation},
 publisher = {Zenodo},
 doi = {10.5281/zenodo.15327950}
}
```
 
## License
This project is licensed under the Creative Commons Attribution 4.0 International License. See the LICENSE file for details.


## Happy coding! 🚀
For questions or support, feel free to open a discussion or an issue.
