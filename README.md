# Statistical Anomaly Detection with Images

This project uses statistical methods to detect anomalous images based on pixel intensity patterns.

## Project Overview

The goal of this project is to determine whether an image is anomalous by analyzing pixel intensity values across a grid of image cells.

Each image is divided into cells, and statistical thresholds are applied to identify unusual patterns.

## Method

- Compute average pixel intensity for each cell across normal images
- Build confidence intervals for each cell
- Compare new images against these intervals
- Flag images as anomalous if values fall outside expected ranges

## Tools

- Python
- NumPy
- SciPy
- OpenCV
- Matplotlib

## What I Learned

This project showed how statistical concepts like confidence intervals can be applied to real-world data. It helped me understand how uncertainty and variation can be used to detect anomalies instead of relying on strict rules.


## Files

- '20110109104734Ch_01.jpeg' - Non-anomalous image
- `20110116090734Lh.jpeg` - anomalous image
- `anomaly_detection.py` - Python Code

