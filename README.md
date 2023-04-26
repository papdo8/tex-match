# tex-match

Command line utility for matching texture maps to the provided template. Supports .png, .jpg, .tif and .bmp formats.

Inspired by receiving 700 textures and spending an afternoon manually sorting out the naming errors.

A collection of test images with various formats and bit depths can be found under /tests/maps.

## How does it work?

The aim of this project was to automate searching for missing texture maps. This can occur if some of the maps belonging together were accidentally misnamed or vertically flipped during format conversion, and normally forces one to manually search through hundreds of images.

The idea was that if a human can recognize matching texture maps based on common shapes and patterns, then there must be a way for a machine to do the same.

The user must specify an input image to be used as the template, and optionally the directory that should be parsed for matches. By default, the program searches the included /tests/maps folder. After loading all images of supported formats, an RGBA to RBG conversion is performed on images with transparency, then everything is converted to grayscale and histogram equalized.

The utility includes two methods for finding matches:

1. Template matching: performs Canny edge detection with Otsu threshold, dilates the edges, then overlaps the template with each parsed image and outputs the resulting maxima. The final output is a list of best matches where the maxima are above a set threshold. Handles possible vertical flipping. Scale invariance is achieved through upscaling low resolution images.

2. Feature matching: performs SIFT feature detection, then compares the keypoint locations and magnitudes between images. When the differences are below the set thresholds, the keypoint is calculated into the final similarity score. The similarity score has the hierarchy of [number of common features] > [mean distance from template keypoint] > [mean magnitude difference from template keypoint]. The final output is a list of best matches sorted by this hierarchy. Scale invariance is achieved through upscaling low resolution images.

## Dependencies

To install dependencies: `$ pip install -r requirements.txt`

## Usage

```
python -m tex_match [-h] [-p PARSE_DIR] [-o OUTPUT_DIR] [-t] [-f] input


positional arguments:
  input                 path to the input texture map

optional arguments:
  -h, --help            show this help message and exit
  -p PARSE_DIR, --parse-dir PARSE_DIR
                        directory to look for matches in
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        directory to write matches to
  -t                    use template matching (fast, default)
  -f                    match based on local features (slow)
```

It is recommended to always run template matching mode unless it is unable to find good matches.

### Examples

Running the search on a .png image located in /tests/maps, in the default template matching mode:

```
python -m tex_match "input.png"
```

Running feature matching mode:

```
python -m tex_match "input.png" -f
```

If the texture maps are located in a different directory:

```
python -m tex_match "path/to/input.png" -p "path/to/texture maps/" -o "path/to/output directory/"
```

If the specified output directory does not exist, it will be automatically created.