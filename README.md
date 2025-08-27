# Demo code for the numerical implementation of the Lippmann-Schwinger formulation for sound scattered by inhomogeneous flows

This repository contains compact Python demos for my algorithm for solving sound propagation in inhomogeneous flows in its Lippmann-Schwinger form, proposed in the paper below. It includes the example cases used in my paper, as well as several related cases that are not included there.


## Citation

T. Li, Y. Zha, and W. Jiang, "[A Fourier Spectral Method for Acoustic Wave
Scattering by Localized Flow Inhomogeneities](https://www.researchgate.net/publication/395108817_A_Fourier_spectral_method_for_acoustic_wave_scattering_by_localized_flow_inhomogeneities),"
*Journal of Theoretical and Computational Acoustics*, vol. 33, 2550013, 2025,
doi: [10.1142/S2591728525500136](https://www.worldscientific.com/doi/10.1142/S2591728525500136).

```bibtex
@article{liFourierSpectralMethod2025,
  title     = {A Fourier Spectral Method for Acoustic Wave Scattering by Localized Flow Inhomogeneities},
  author    = {Li, Tianyu and Zha, Yang and Jiang, Weikang},
  year      = {2025},
  month     = aug,
  journal   = {Journal of Theoretical and Computational Acoustics},
  volume    = {33},
  pages     = {2550013},
  publisher = {World Scientific Publishing Co.},
  issn      = {2591-7285},
  doi       = {10.1142/S2591728525500136}
}
```

## Contents

- `vortex2dscatter.py`: Fourier-collocation solvers for the complete 2D scattering equation, the $\mathcal{O}(\text{Ma})$ approximation, and the high-frequency paraxial approximations.
- `clairgabard_ref.py`: reimplementation of a radial RK4 reference solver from [Clair et al. (2018)](https://doi.org/10.1017/jfm.2018.94).

- `01-taylor-vortex.ipynb`: results for plane waves scattered by a Taylor vortex.
  - `02-bornrytov-approx.ipynb`: assessment of the Born and Rytov approximations for this problem.
  - `03-paraxial-approx.ipynb`: assessment of a paraxial-like slowly varying amplitude approximation for this problem.
- `04-jet-beam.ipynb`: results for beams diffracted by a jet.
- `05-rotating-vorticies.ipynb`: results for spectral broadening of plane waves scattered by a rotating vortex pair.

## Kernel truncation method

For implementation details of the kernel truncation method used for the fast convolution in these demos, see also [ltyylt/demo-ktmhhd](https://github.com/ltyylt/demo-ktmhhd).