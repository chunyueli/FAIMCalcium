## FAIMCalcium
Fully affine invariant methods for the field-of-view alignment of calcium images across multiple imaging sessions.

## Notes
This code includes the Cellpose package written by Carsen Stringer and Marius Pachitariu. If you want to know more about Cellpose, please visit https://github.com/MouseLand/cellpose.

## System requirements
This code has been seriously tested on Windows 10 and Ubuntu 18.04. Codes requires Python 3.7.

## Installation
1. Install environment by running: conda env create -f [your file path to FAIM_package/]environment.yaml
2. Enter environment by running: conda activate FAIMCalcium

## Usage
* cd [path to the upper level of the AffineCa2p]
* run python
* from AffineCa2p.FAIM import FAIMCaSig
* Type: Tmatrices, regImages, regROIs=FAIMCaSig.AlignIm("path to/examples/A5") or Tmatrices, regImages, regROIs=FAIMCaSig.AlignIm("path to/examples/A6"), then the command will automatically register all the FOV images under the folder A5 or A6. Besides, the code will automatically extract ROI masks using the cellpose and save the masks under the folder of FOV images ("A5" or "A6").
* If the ROI masks have already been derived, providing a path to the ROI masks folder can save time. Run Tmatrices, regImages, regROIs=FAIMCaSig.AlignIm("path_to_FOV_images", "path_to_ROI_masks")
* Try different affine methods and set different template indexes, by running: Tmatrices, regImages, regROIs=FAIMCaSig.AlignIm("path_to_FOV_images", "path_to_ROI_masks", templateID=index_number, method='surf')

## Outputs:
* Tmatrices: list of the transformation matrices
* regImages: list of the registered FOV images
* regROIs: list of the registered ROI masks
Besides, the Tmatrices and the registered FOV images as well as the registered ROIs masks will also be saved as a csv file and an image under the folder of FOV images.

## function parameters
* path_to_FOV: path of the folder containing FOV images
* path_to_masks: path of the folder containing ROIs mask. If the value is empty, the code will automatically extract ROI masks using the cellpose and save the masks under the folder of FOV images ("A5" or "A6"). If the ROI masks have already obtained, providing the path to the ROI masks folder can save time.
* preprocess: whether to apply contrast adjustment to the original FOV images. The default value is False.
* diameter: neuron diameter. If the value is None, the diameter will be automatically estimated by Cellpose. Otherwise, the neuron mask will be detected based on the given diameter value by Cellpose.
* templateID: choose which FOV image as a template for alignment.  Its default value is zero, indicating the first FOV image.
* iterNum: the number of iterations of the fully affine invariant method(default: 100).
* method: name of the fully affine invariant method (default ['sift']). Name of the candidate methods:
  * 'sift' corresponds to 'ASIFT'
  * 'surf' corresponds to 'ASURF'
  * 'akaze' corresponds to 'AAKAZE'
  * 'brisk' corresponds to 'ABRISK'
  * 'orb' corresponds to 'AORB'

## Common Issues
If you receive the error: "opencv2/nonfree/features2d.hpp": No such file or directory, please try following commands:
pip uninstall opencv-python
pip install Opencv-contrib-python==3.4.2.17

## Dependencies
* python=3.7.4
* mkl=2019.3
* numpy
* scikit-image
* numba>=0.43.1
* pyqt
* scipy
* matplotlib
* mxnet-mkl
* pandas
* pyqtgraph==0.11.0rc0
* natsort
* google-cloud-storage
* tqdm
* Opencv-contrib-python==3.4.2.17
