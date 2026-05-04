# E57 Image Extractor

This script extracts embedded 2D images from an E57 file and writes a JSON metadata file for each image.

## What the script does

Given an E57 scan file, `extract.py`:

1. Opens the file with `pye57`.
2. Reads the `images2D` section from the E57 root.
3. For each image entry:
   - Reads the JPEG bytes from either the `pinholeRepresentation` or `sphericalRepresentation`.
   - Decodes the JPEG with OpenCV.
   - Saves the image as `images/<image_name>.jpg`.
   - Saves metadata as `metadata/<image_name>.json`.
4. If pose data is present, the metadata includes:
   - `translation`: `x`, `y`, `z`
   - `rotation`: quaternion `x`, `y`, `z`, `w`

If the file contains no `images2D` entries, the script logs a warning and exits without creating output files.

## Requirements

- Python 3.9+
- `numpy`
- `opencv-python`
- `pye57`

Install dependencies with:

```bash
pip install numpy opencv-python pye57
```

To generate cubemaps from extracted panoramas, the same `numpy` and `opencv-python` dependencies are required.

## Usage

Basic usage:

```bash
python extract.py Scans.e57
```

Specify a custom output folder:

```bash
python extract.py Scans.e57 --outfolder extracted_data
```

Generate cubemap faces from the extracted panoramas:

```bash
python cubemap.py ext/Scans/images
```

Specify a custom cubemap output folder:

```bash
python cubemap.py ext/Scans/images --outfolder other_cubemaps
```

## Output structure

The extractor creates this folder layout:

```text
ext/
  Scans/
    images/
      image_1.jpg
      image_2.jpg
    metadata/
      image_1.json
      image_2.json
```

The cubemap script creates this folder layout:

```text
ext/
  Scans/
    images/
      p000016.jpg
      cubemaps/
        p000016/
          posx.jpg
          negx.jpg
          posy.jpg
          negy.jpg
          posz.jpg
          negz.jpg
```

Example metadata with pose:

```json
{
    "name": "image_1",
    "translation": {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0
    },
    "rotation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "w": 1.0
    }
}
```

Example metadata without pose:

```json
{
    "name": "image_1"
}
```

## Notes and limitations

- The script assumes each embedded image is stored as JPEG data.
- Output image filenames are based directly on the E57 image `name`.
- The script does not currently sanitize filenames or handle duplicate image names.
- If decoding fails, `cv2.imwrite` may receive `None` and fail depending on the input data.
- `extract.py` now stores outputs under `ext/<e57_filename>/...`, for example `ext/Scans/images` and `ext/Scans/metadata`.
- `cubemap.py` stores cubemaps by default under the input images folder, for example `ext/Scans/images/cubemaps`.
- `cubemap.py` expects equirectangular panoramas and uses the generic cubemap face order `posx`, `negx`, `posy`, `negy`, `posz`, `negz`.
- For valid 2:1 panoramas, cubemap face resolution is the highest possible without upscaling: `min(width // 4, height // 2)`.
- For the current `7680x3840` panoramas, that means `1920x1920` cubemap faces.
- Non-2:1 inputs are converted in best-effort mode by center-cropping to the largest 2:1 region before conversion, which may be geometrically inexact.

## Running tests

The unit tests use mocks and do not require a real E57 file.

```bash
python -m unittest -v
```
