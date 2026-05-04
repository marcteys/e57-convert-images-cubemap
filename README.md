# E57 Image and Cubemap Converter

This project extracts embedded 2D images from an E57 file and converts panoramic images into cubemap faces.

## What the project does

`extract.py`:

1. Opens an E57 file with `pye57`.
2. Reads the `images2D` section.
3. Extracts each embedded JPEG image.
4. Saves images and JSON metadata under `ext/<e57_filename>/...`.
5. Preserves the original E57 image name in metadata and generates unique output filenames when names are duplicated.

`cubemap.py`:

1. Reads extracted panorama images from a folder such as `ext/Scans/images`.
2. Detects whether each image is already a valid 2:1 panorama.
3. Converts each panorama into 6 cubemap faces.
4. Saves cubemaps under `ext/<e57_filename>/images/cubemaps/<image_name>/`.

## Requirements

- Python 3.9+
- `numpy`
- `opencv-python`
- `pye57`

Install dependencies with:

```bash
pip install numpy opencv-python pye57
```

## Usage

Extract images and metadata from an E57 file:

```bash
python extract.py Scans.e57
```

Extract to a custom root output folder:

```bash
python extract.py Scans.e57 --outfolder ext
```

Generate cubemaps from extracted images:

```bash
python cubemap.py ext/Scans/images
```

Generate cubemaps into a custom output folder:

```bash
python cubemap.py ext/Scans/images --outfolder other_cubemaps
```

## Output structure

After extraction:

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

After cubemap generation:

```text
ext/
  Scans/
    images/
      image_1.jpg
      image_2.jpg
      cubemaps/
        image_1/
          posx.jpg
          negx.jpg
          posy.jpg
          negy.jpg
          posz.jpg
          negz.jpg
        image_2/
          posx.jpg
          negx.jpg
          posy.jpg
          negy.jpg
          posz.jpg
          negz.jpg
```

## Metadata format

Example metadata with pose:

```json
{
    "name": "Panorama",
    "output_name": "Panorama_002",
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
    "name": "Panorama",
    "output_name": "Panorama"
}
```

## Cubemap behavior

- Face names are `posx`, `negx`, `posy`, `negy`, `posz`, `negz`.
- Cubemaps are generated with the highest resolution possible without upscaling.
- Face size is computed as `min(width // 4, height // 2)` for valid 2:1 panoramas.
- For a `7680x3840` panorama, face size is `1920x1920`.
- For non-2:1 images, the script performs a centered best-effort crop to the largest possible 2:1 region before conversion.

## Notes and limitations

- The extractor assumes embedded images are stored as JPEG data.
- Output filenames are based on the E57 image name.
- If multiple E57 images share the same name, the extractor appends suffixes like `_002`, `_003`, and so on.
- `cubemap.py` is intended for equirectangular panoramas.
- If an image cannot be decoded, it is skipped during cubemap generation.

## Running tests

The unit tests use mocks and do not require a real E57 file.

```bash
python -m unittest -v
```
