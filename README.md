Code to generate, plot and fit data used in Manuscript "Deviations from spectral dirac comb in Rayleigh-Benard turbulent bursting".


### Raw data of $K$ 

The raw data of the energy integral $K$ and the according time values are available in `RB_data/`. The values `1e-4` and `1.6e-3` refer to the diffusivity $\kappa$ and viscosity $\mu$.

### Reproducing figures

To reproduce the exact conda environment used to produced figures use the included `Periodic-pulses-paper.yml` file:
```console
conda env create -f Periodic-pulses-paper.yml
```
Run the scripts `spectra_1_6e-3.py` and `spectra_1e-4.py` in order to create figure 1 and 8. If you want to plot the figures without the fit, comment out the lines 31 and 42 of the two scripts. The remaining figures are created by the `crreate_figure_*.py` scripts. 
