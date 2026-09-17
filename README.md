# Pilot-Jacobian-Speckle-Spectrometer-
We presents a self-calibrating mm-fiber speckle spectrometer that uses time-gated reference pilots and pre-calibrated environmental Jacobians to compensate for temperature- and bend-induced modal drift. Proposed approach updates the tx matrix without repeated λ sweeps, enabling more stable spectral reconstruction under dynamic env. perturbations.

## Principal model settings

- Wavelength grid: 1546–1554 nm, 81 channels (0.1 nm spacing)
- Sample example band: 1547–1553 nm
- Pilots: 1546.5 and 1553.5 nm
- MMF: 2 m, 105-µm core, NA 0.22
- Effective modal basis: 42 modes
- Camera ROI: 12×12 = 144 samples
- Nominal sample/pilot SNR: 35/40 dB
- Finite-difference calibration increments: 0.05 °C and 0.01 bend-state unit
- Regularization: alpha = 2e-6; beta = 2e-7
- Seed used by the reference Python study: 20260913

## Reference manuscript metrics

- cond(G): 6.008286821
- Example stale error: 0.403283
- Example single-pilot error: 0.242033
- Example proposed error: 0.138486
- Example oracle error: 0.034913
- Dynamic mean stale error: 0.257455
- Dynamic mean proposed error: 0.066113


