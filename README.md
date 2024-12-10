# Deviations from spectral Dirac comb in Rayleigh Bénard turbulent bursting

Code to generate, plot and fit data used in Manuscript *"Deviations from spectral dirac comb in Rayleigh-Benard turbulent bursting"*.

## Reproducing the environment

To set up the environment to reproduce the figures, clone the repo to your local machine, then install the project and its dependencies with poetry:

```sh
git clone https://github.com/uit-cosmo/cosmoplots.git
cd cosmoplots
poetry install
```

## Description of figure generation

The top-level scripts may be run without any extra preparation.

### create_figure_gamma_wait.py

Power spectral density and autocorrelation of the stochastic process with exponentially distributed amplitudes and Gamma distributed waiting times.

Generates figure 2 in the manuscript.

### `create_figure_exp_lap_amp.py`

Power spectral density and autocorrelation of the stochastic process with exponentially and symmetrically Laplace distributed amplitudes.

Generates figure 3 in the manuscript.

### `create_figure asym_lap_amp.py`

Power spectral density and autocorrelation of the stochastic process with asymmetrically Laplace distributed amplitudes.

Generates figure 4 in the manuscript.

### Raw data of $K$

The raw data of the energy integral $K$ and the according time values are available in `RB_data/`. The values `1e-4` and `1.6e-3` refer to the diffusivity $\kappa$ and viscosity $\mu$.

### Reproducing figures

To reproduce the exact conda environment used to produced figures use the included `Periodic-pulses-paper.yml` file:

```console
conda env create -f Periodic-pulses-paper.yml
```

Run the scripts `spectra_1_6e-3.py` and `spectra_1e-4.py` in order to create figure 1 and 8. If you want to plot the figures without the fit, comment out the lines 31 and 42 of the two scripts. The remaining figures are created by the `create_figure_*.py` scripts.

### Run Rayleigh-Benard model in BOUT++

If you prefer to run the RB-model from scratch in BOUT++ you find all necessary files in `BOUT_files`. The `PhysicsModel` is defined in `rb-model.cxx` and the simulation inputs, such as $\kappa$ and $\mu$, are defined in `BOUT.inp`. The data shown in the paper is created with BOUT++ version 4.4.0. Check the BOUT++ manual for instructions for to install BOUT++ and run a custom `PhysicsModel`: <https://bout-dev.readthedocs.io/en/stable/>

You can calculate $K$ from the simulation output using the `BOUT_files/calculate_K.py` script. For this, install the `xbout` package (<https://github.com/boutproject/xBOUT>) and adjust the path to the BOUT++ output data in line 4.
