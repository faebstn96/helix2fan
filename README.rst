.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
.. image:: https://img.shields.io/badge/arXiv-2201.10345-f9f107.svg
    :target: https://arxiv.org/abs/2201.10345

==========================================
Helix2Fan: Helical to fan-beam CT geometry rebinning and differentiable reconstruction of DICOM-CT-PD projections
==========================================

This repository provides code to load raw helical DICOM-CT-PD CT projections and
rebin them to flat detector fan-beam geometry. We follow the algorithm
of `Noo et al. <https://doi.org/10.1088/0031-9155/44/2/019>`__ to rebin projections acquired on a
helical CT trajectory to a circular trajectory. This enables CT reconstructing using conventional reconstruction
frameworks and even differentiable filtered back projection (FBP) operators.
In addition to the rebinning framework we provide code to reconstruct the processed projections using a differentiable
FBP operator from the `torch-radon <https://github.com/matteo-ronchetti/torch-radon>`__
framework. This differentiable operator allows propagating a loss metric, calculated on the reconstructed image,
back to the projection data. It, therefore, enables intervening into the reconstruction pipeline at different stages
with, e.g., neural networks.

In our associated paper `On the benefit of dual-domain denoising in a self-supervised low-dose CT setting <xyz>`__
we use this framework to render medical low-dose CT data acquired on a helical trajectory possible for end-to-end
reconstruction and denoising in both projection and image domain using neural networks. Please refer to our
`arXiv <https://arxiv.org/pdf/xyz.pdf>`__ publication if you find our code useful.

Projection data (DICOM-CT-PD):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our data loader supports the standardized `DICOM-CT-PD format <https://doi.org/10.1118/1.4935406>`__ for
loading and preparing the projection data and geometry information for rebinning and CT reconstruction.
The largest public low-dose CT image and projection data set
`LDCT-and-Projection-data <https://doi.org/10.1002/mp.14594>`__ provides projection data of more than 100
abdomen/chest/head CT scans in the DICOM-CT-PD format. Please download projection data
from `their repository <https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026>`__ to run
the rebinning in this framework.

Setup:
~~~~~~

1. Create and activate a python environment (python>=3.7).
2. Install `Torch <https://pytorch.org/get-started/locally/>`__.
3. Download the DICOM-CT-PD projection data from `TCIA <https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026>`__.
4. Run the rebinning code:

.. code:: bash

   python main.py --path_dicom /path/to/DICOM-CT-PD/data/folder

5. Find the rebinned data in the out folder (default).

If you want to run the reconstruction script you additionally need to install torch-radon

1. Install `torch-radon <https://github.com/matteo-ronchetti/torch-radon>`__

.. code:: bash

   git clone https://github.com/matteo-ronchetti/torch-radon.git
   cd torch-radon
   python setup.py install

2. The current torch-radon repository uses some outdated PyTorch functions. You can use the torch-radon_fix.patch in the torch-radon_fix folder of this repository to fix this problem:

.. code:: bash

   cd helix2fan
   git apply torch-radon_fix/torch-radon_fix.patch

3. Run the reconstruction code using the rebinned data:

.. code:: bash

   python reco_example_fan_beam.py --path_proj /path/to/fan-beam/projections.tif

4. Find the reconstructed CT images in the out folder (default).


Example scripts:
~~~~~~~~~~~~~~~~
-  Please use the main.py for projection loading and rebinning.
-  Please use reco_example_fan_beam.py to reconstruct the final CT image from the rebinned projection data.

Citation:
~~~~~~~~~

If you find our code useful, please cite our work

::

   @article{wagner2022dual,
     title={On the benefit of dual-domain denoising in a self-supervised low-dose CT setting},
     author={Wagner, Fabian and Thies, Mareike and Pfaff, Laura and Aust, Oliver and Pechmann, Sabrina and Maul, Noah and Rohleder, Maximilian and Gu, Mingxuan and Utz, Jonas and Denzinger, Felix and Maier, Andreas},
     journal={arXiv preprint arXiv:xyz},
     year={2022},
     doi={https://arxiv.org/xyz}
   }


Troubleshooting
~~~~~~~~~~~~~~~
