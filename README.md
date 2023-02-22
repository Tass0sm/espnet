<div align="left"><img src="doc/image/espnet_logo1.png" width="550"/></div>

Our fork of espnet for CSE 5539 experiments.

We explored using ideas from image processing on the spectogram. 

1. Using axial attention blocks
2. Using SWIN transformer blocks

# ASR

## Baseline

espnet_model with transformer encoder / decoder

## Variant 1

Processing 2d frames of the spectogram without prior convolutions. axial
attention on frames.

Result: less effective

## Variant 2

