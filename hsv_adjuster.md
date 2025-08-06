# HSV Adjuster

This script is a tool for adjusting HSV color thresholds for image processing. It uses OpenCV to create trackbars for adjusting the Hue, Saturation, and Value (HSV) ranges. The script captures video from a camera, applies a mask based on the HSV thresholds, and displays the original and masked images.

## Usage

Run the script from the command line:

```bash
python hsv_adjuster.py
```

A window will appear with trackbars for adjusting the lower and upper HSV threshold values. The script will print the current threshold values to the console whenever they are changed.

