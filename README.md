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

### [rayleigh_benard](rayleigh_benard)

Time series, power spectral density and conditional averaging of the Rayleigh Bénard data. [create_figure.py](rayleigh_benard/create_figure.py) may be run directly. See the section below for simulation details.

Generates figures 1, 9 and 10 in the manuscript.

Note: In the manuscript, the RB kinetic energy has changed name from $\mathcal{K}$ to $\mathcal{E}$. This name change may not be completely reflected in the code.

### [create_figure_gamma_wait.py](create_figure_gamma_wait.py)

Power spectral density and autocorrelation of the stochastic process with exponentially distributed amplitudes and Gamma distributed waiting times.

Generates figure 2 in the manuscript.

### [create_figure_exp_lap_amp.py](create_figure_exp_lap_amp.py)

Power spectral density and autocorrelation of the stochastic process with exponentially and symmetrically Laplace distributed amplitudes.

Generates figure 3 in the manuscript.

### [create_figure_asym_lap_amp.py](create_figure_asym_lap_amp.py)

Power spectral density and autocorrelation of the stochastic process with asymmetrically Laplace distributed amplitudes.

Generates figure 4 in the manuscript.

### [create_figure_gaussian_jitter_wait.py](create_figure_gaussian_jitter_wait.py)

Power spectral density and autocorrelation of the stochastic process with periodic arrivals with Gaussian jitter.

Generates figure 5 in the manuscript.

### [create_figure_gaussian_waiting_times.py](create_figure_gaussian_waiting_times.py)

Power spectral density and autocorrelation of the stochastic process with Gaussian waiting times.

Generates figure 6 in the manuscript.

### [wait_compare](wait_compare)

Compares the parts due to renewal waiting times in the PSD for normal, Gamma and inverse gamma waiting times.

Run [gen_gammainv_cf.py](wait_compare/gen_gammainv_cf.py) and [gen_norm_wait_num_psd.py](wait_compare/gen_norm_wait_num_psd.py) first, then [create_figure.py](wait_compare/create_figure.py).

Generates figure 7 in the manuscript.

### [jitter_numeric_vs_analytic](jitter_numeric_vs_analytic)

Compares the parts due to jittered waiting times in the PSD for normal jitter times using the analytic and numeric solutions.

Run [gen_norm_jitter_num_psd.py](jitter_numeric_vs_analytic/gen_norm_jitter_num_psd.py) first, then [create_figure.py](jitter_numeric_vs_analytic/create_figure.py).

Generates figure 8 in the manuscript.

## Generating Rayleigh-Bénard data

### Raw data of $K$

The raw data of the energy integral $K$ and the according time values are available in [RB_data](rayleigh_benard/RB_data). The values `1e-4` and `1.6e-3` refer to the diffusivity $\kappa$ and viscosity $\mu$.

### Run Rayleigh-Benard model in BOUT++

If you prefer to run the RB-model from scratch in BOUT++ you find all necessary files in [BOUT_files](rayleigh_benard/BOUT_files). The `PhysicsModel` is defined in [rb-model.cxx](rayleigh_benard/BOUT_files/rb-model.cxx) and the simulation inputs, such as $\kappa$ and $\mu$, are defined in [BOUT.inp](rayleigh_benard/BOUT_files/BOUT.inp). The data shown in the paper is created with BOUT++ version 4.4.0. Check the BOUT++ manual for instructions for to install BOUT++ and run a custom `PhysicsModel`: <https://bout-dev.readthedocs.io/en/stable/>

You can calculate $\mathcal{E}$ from the simulation output using the [calculate_K.py](rayleigh_benard/BOUT_files/calculate_K.py) script. For this, install the `xbout` package (<https://github.com/boutproject/xBOUT>) and adjust the path to the BOUT++ output data in line 4.
