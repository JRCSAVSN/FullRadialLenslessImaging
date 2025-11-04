# Toward all-in-focus lensless imaging with full-aperture radial masks (Optics Express 2025)

---


## Abstract (Summary)

---
In this work we show that radial patterns enable wider depth-of-field lensless imaging, while also enabling efficient modeling of the lensless imaging system with full-aperture masks. This allows us to capture longer continuous scenes than previous approaches, with a competitive inference time and reconstruction quality.

![image](data/figs/OE_2025_psfcorr.pdf)

## Path to data

---
The data for each of the experiments on the paper are as follows:

* Section 4 (Analysis of PSF depth dependence from correlation): ```./data/psf/*```
* Section 5 (MNIST digit reconstruction): [Link](https://drive.google.com/drive/folders/1A4Sk2m-QhhYObge7z7N8gauiP7tVqu-L?usp=sharing)
* Section 6: (QR codes reconstruction): [Coming soon]()
* Section 7: (Artificially-extended convolution experiments): ```./data/monitor/*```, ```./data/two_pawns_sanken_multiexposureV2/*``` and ```./data/psfV2/*```

## Code Usage

---
The experiments for the artificially-extended PSF deconvolution with our full-aperture mask can be replicated by running the .sh files as:
```bash
./big_conv_prototype.sh
./big_conv_simulation.sh
```
Results will be saved in a new folder ```./results/*``` for both prototype camera and simulated experiments.

## Citation

---
Citation will be appropriately filled when the manuscript is published (currently accepted but not published)
```
@article{Neto:25,
author = {Jose Reinaldo Cunha Santos A V Silva Neto and Hodaka Kawachi and Yasushi Yagi and Tomoya Nakamura},
journal = {Opt. Express},
number = {},
volume = {],
pages = {},
year = {2025},
url = {}
} 
```