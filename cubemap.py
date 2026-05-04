import argparse
import logging
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FACE_NAMES = ("posx", "negx", "posy", "negy", "posz", "negz")


def require_dependencies():
    missing = []
    if np is None:
        missing.append("numpy")
    if cv2 is None:
        missing.append("opencv-python")
    if missing:
        missing_list = ", ".join(missing)
        raise ModuleNotFoundError(
            f"Missing required dependencies: {missing_list}. "
            f"Install them with `pip install {missing_list}`."
        )


def compute_face_size(width, height):
    face_size = min(width // 4, height // 2)
    if face_size < 1:
        raise ValueError(f"Image is too small to generate a cubemap: {width}x{height}")
    return face_size


def normalize_to_2_to_1(image):
    height, width = image.shape[:2]
    if width < 2 or height < 1:
        raise ValueError(f"Image is too small to normalize: {width}x{height}")

    if width == height * 2:
        return image, False

    if width > height * 2:
        target_width = height * 2
        x_offset = (width - target_width) // 2
        return image[:, x_offset : x_offset + target_width], True

    target_height = max(1, width // 2)
    y_offset = max(0, (height - target_height) // 2)
    return image[y_offset : y_offset + target_height, :], True


def _face_vectors(face_name, uu, vv):
    if face_name == "posx":
        return np.ones_like(uu), vv, -uu
    if face_name == "negx":
        return -np.ones_like(uu), vv, uu
    if face_name == "posy":
        return uu, np.ones_like(uu), -vv
    if face_name == "negy":
        return uu, -np.ones_like(uu), vv
    if face_name == "posz":
        return uu, vv, np.ones_like(uu)
    if face_name == "negz":
        return -uu, vv, -np.ones_like(uu)
    raise ValueError(f"Unsupported face name: {face_name}")


def _render_face(image, face_name, face_size):
    require_dependencies()

    height, width = image.shape[:2]
    x_coords = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
    y_coords = np.linspace(1.0, -1.0, face_size, dtype=np.float32)
    uu, vv = np.meshgrid(x_coords, y_coords)

    x, y, z = _face_vectors(face_name, uu, vv)
    norm = np.sqrt(x * x + y * y + z * z)
    x /= norm
    y /= norm
    z /= norm

    theta = np.arctan2(x, z)
    phi = np.arcsin(np.clip(y, -1.0, 1.0))

    map_x = ((theta / (2.0 * np.pi)) + 0.5) * width
    map_y = (0.5 - (phi / np.pi)) * height

    map_x = np.mod(map_x, width).astype(np.float32)
    map_y = np.clip(map_y, 0, height - 1).astype(np.float32)

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def generate_cubemap_faces(image, face_size):
    return [(face_name, _render_face(image, face_name, face_size)) for face_name in FACE_NAMES]


def load_image(image_path):
    require_dependencies()
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def save_face_image(output_path, image, quality):
    require_dependencies()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(output_path),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise IOError(f"Failed to save cubemap face: {output_path}")


def default_cubemap_output_folder(input_folder):
    return Path(input_folder) / "cubemaps"


def convert_image_to_cubemap(image_path, output_folder, quality=95):
    logging.info("Processing panorama %s...", image_path.name)
    image = load_image(image_path)
    if image is None:
        logging.warning("Skipping unreadable image: %s", image_path)
        return False

    working_image, best_effort = normalize_to_2_to_1(image)
    if best_effort:
        logging.warning(
            "Image %s is not a 2:1 panorama. Applying a centered best-effort crop before conversion.",
            image_path.name,
        )

    face_size = compute_face_size(working_image.shape[1], working_image.shape[0])
    faces = generate_cubemap_faces(working_image, face_size)

    image_output_folder = output_folder / image_path.stem
    for face_name, face_image in faces:
        output_path = image_output_folder / f"{face_name}.jpg"
        save_face_image(output_path, face_image, quality)

    logging.info(
        "Saved %d cubemap faces for %s at %dx%d.",
        len(faces),
        image_path.name,
        face_size,
        face_size,
    )
    return True


def iter_input_images(input_folder, extension):
    normalized_extension = extension.lower()
    if not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"

    return sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() == normalized_extension
    )


def convert_folder(input_folder, output_folder=None, extension=".jpg", quality=95):
    input_path = Path(input_folder)
    output_path = default_cubemap_output_folder(input_path) if output_folder is None else Path(output_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_path}")

    image_paths = iter_input_images(input_path, extension)
    if not image_paths:
        logging.warning("No input images found in %s with extension %s.", input_path, extension)
        return {"processed": 0, "skipped": 0}

    output_path.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for image_path in image_paths:
        try:
            if convert_image_to_cubemap(image_path, output_path, quality=quality):
                processed += 1
            else:
                skipped += 1
        except Exception as exc:
            skipped += 1
            logging.warning("Failed to convert %s: %s", image_path.name, exc)

    logging.info(
        "Cubemap conversion complete. Processed=%d, Skipped=%d, Output=%s",
        processed,
        skipped,
        output_path,
    )
    return {"processed": processed, "skipped": skipped}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert extracted equirectangular panoramas into cubemap faces."
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default="ext/Scans/images",
        help="Folder containing extracted panorama images",
    )
    parser.add_argument(
        "--outfolder",
        default=None,
        help="Output folder for generated cubemap face folders",
    )
    parser.add_argument(
        "--extension",
        default=".jpg",
        help="File extension to read from the input folder",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for saved cubemap faces (0-100)",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert_folder(args.input_folder, args.outfolder, extension=args.extension, quality=args.quality)
