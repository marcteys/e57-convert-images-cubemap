import numpy as np
import cv2
import pye57
import argparse
import json
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_named_child(node, child_name):
    if isinstance(node, dict):
        return node.get(child_name)

    try:
        return node[child_name]
    except Exception:
        return None


def iter_child_names(node):
    if isinstance(node, dict):
        return list(node.keys())

    names = []
    try:
        for child in node:
            element_name = getattr(child, "elementName", None)
            if callable(element_name):
                names.append(element_name())
    except TypeError:
        return names

    return names


def build_output_base_folder(e57_path, output_root):
    e57_name = Path(e57_path).stem
    return Path(output_root) / e57_name


def build_unique_output_name(image_name, used_names):
    count = used_names.get(image_name, 0) + 1
    used_names[image_name] = count
    if count == 1:
        return image_name
    return f"{image_name}_{count:03d}"


def get_image_representation(image2D):
    preferred_names = (
        "pinholeRepresentation",
        "sphericalRepresentation",
        "cylindricalRepresentation",
        "visualReferenceRepresentation",
    )
    for field_name in preferred_names:
        child = get_named_child(image2D, field_name)
        if child is not None:
            return child

    for field_name in iter_child_names(image2D):
        if not str(field_name).endswith("Representation"):
            continue
        child = get_named_child(image2D, field_name)
        if child is not None:
            return child

    return None


def decode_embedded_image(image_representation):
    for field_name in ("jpegImage", "pngImage"):
        encoded_image = get_named_child(image_representation, field_name)
        if encoded_image is None:
            continue
        encoded_image_data = np.zeros(shape=encoded_image.byteCount(), dtype=np.uint8)
        encoded_image.read(encoded_image_data, 0, encoded_image.byteCount())

        image = cv2.imdecode(encoded_image_data, cv2.IMREAD_COLOR)
        if image is not None:
            return image, field_name

    return None, None


def extract_and_save_images_and_metadata(e57_path, output_root):
    logging.info("Loading E57 file...")
    e57 = pye57.E57(e57_path)
    imf = e57.image_file
    root = imf.root()

    logging.info("File loaded successfully.")

    if not root['images2D']:
        logging.warning("File contains no 2D images. Exiting...")
        return

    output_base_folder = build_output_base_folder(e57_path, output_root)
    images_output_folder = output_base_folder / "images"
    metadata_output_folder = output_base_folder / "metadata"
    images_output_folder.mkdir(parents=True, exist_ok=True)
    metadata_output_folder.mkdir(parents=True, exist_ok=True)
    used_names = {}

    for image_idx, image2D in enumerate(root['images2D']):
        logging.info(f"Processing image {image_idx}...")

        image_name = str(image2D['name'].value())
        output_name = build_unique_output_name(image_name, used_names)
        image_representation = get_image_representation(image2D)
        if image_representation is None:
            logging.warning(
                "Skipping image %s because it has no supported image representation.",
                output_name,
            )
            continue

        image, image_format = decode_embedded_image(image_representation)
        if image is None:
            logging.warning(
                "Skipping image %s because OpenCV could not decode its embedded image data.",
                output_name,
            )
            continue

        image_path = images_output_folder / f"{output_name}.jpg"
        cv2.imwrite(str(image_path), image)
        logging.info("Saved %s to %s", image_format, image_path)

        pose = get_named_child(image2D, "pose")
        if pose is not None:
            translation = pose['translation']
            rotation = pose['rotation']
            x = float(translation['x'].value())
            y = float(translation['y'].value())
            z = float(translation['z'].value())
            rx = float(rotation['x'].value())
            ry = float(rotation['y'].value())
            rz = float(rotation['z'].value())
            rw = float(rotation['w'].value())
            logging.info(f"Coords: x={x}, y={y}, z={z}, rx={rx}, ry={ry}, rz={rz}, rw={rw}")

            # Save pose information in metadata
            metadata = {
                'name': image_name,
                'output_name': output_name,
                'translation': {'x': x, 'y': y, 'z': z},
                'rotation': {'x': rx, 'y': ry, 'z': rz, 'w': rw}
            }
        else:
            metadata = {'name': image_name, 'output_name': output_name}
            

        metadata_path = metadata_output_folder / f"{output_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Saved metadata to {metadata_path}")

    logging.info("All images and metadata processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save 2D images and metadata from an E57 file.')
    parser.add_argument('e57', help='Path to the E57 file')
    parser.add_argument('--outfolder', default='ext', help='Root output folder for extracted data')
    args = parser.parse_args()

    extract_and_save_images_and_metadata(args.e57, args.outfolder)
