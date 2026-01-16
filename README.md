# Image analysis scripts for PhD thesis

This repository contains image analysis pipelines used to quantify cellular and tissue-level features in the experiments described in Heide F. B. Murray's PhD thesis from the Harvard department of Chemical Biology, 2026. 

## Overview
The scripts implement standard image-processing workflows for:
- quantifying pixel intensities of tissue slice images at a certain distance from a specified region of interest
- creating a mask from a control channel and applying to a channel for quantification
- nuclei and puncta counting
- output of graphs via matplotlib or output of quantified data in format compatible with GraphPad prism for plotting and statistical analysis

No novel image-analysis algorithms are introduced.

## Scripts
- `chromogenic_IHC-2.ipynb`
  Inputs .csv files output from ZEN blue software that measure grayscale intensity (along with R, G, B intensity) from edge to edge of DAB-stained organoid slice. 
  In total, there are four .csv files per organoid, 4-9 organoids per stain, 2 treatment conditions (DMSO vs. 25uM BL-918), and four cell lines.
  Representative images of organoid slices quantified are shown in 4.10-4.13. 
  Used for Figures 4.10, 4.11, 4.12. 
  
- `mask_and_quant.py`  
  Creates mask based on user-defined thresholding value of cell body stain image (ie, MAP2 or phalloidin) and applies this mask to the same well/field image in a separate channel of interest to be quantified.
  Essentially, quantifies intensity of the channel of interest within the bounds of the cell, effectively excluding quantfication of the background.  
  Used for Figures 3.2, 5.3.

- `count_puncta_nuclei.py`  
  Counts nuclei objects in DAPI channel. Counts puncta objects (using user-defined threshold) in specified channel of interest.
  Ouputs various metrics about nuclei and puncta (including: # nuclei, # puncta, puncta / nuclei, puncta raw integrated density, etc) to .csv format to be plotted and statistcally tested in GraphPad prism.
  Used for Figures 3.3, 3.9, 3.10, 5.4, 5.5, 5.7.

## Requirements
- Python 3.9
- glob
- pandas
- numpy
- matplotlib
- statistics
- scipy
- math
- os
- skimage
- PIL
- imageio
- logging

## Reproducibility
This repository is archived as a permanent release corresponding to the version used in the thesis. See the DOI referenced in the thesis for the exact version.
