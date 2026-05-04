import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "extract.py"


class FakeValue:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class FakeBlob:
    def __init__(self, payload):
        self.payload = payload

    def byteCount(self):
        return len(self.payload)

    def read(self, buffer, start, count):
        for index, value in enumerate(self.payload[start : start + count]):
            buffer[index] = value


class FakeImageFile:
    def __init__(self, root_data):
        self._root_data = root_data

    def root(self):
        return self._root_data


class FakeE57File:
    def __init__(self, root_data):
        self.image_file = FakeImageFile(root_data)


def load_extract_module(root_data, imdecode_result="decoded-image"):
    fake_numpy = types.SimpleNamespace(
        uint8="uint8",
        zeros=lambda shape, dtype=None: bytearray(shape),
    )

    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        written_files=[],
    )

    def imdecode(data, flag):
        fake_cv2.last_imdecode = (bytes(data), flag)
        return imdecode_result

    def imwrite(path, image):
        fake_cv2.written_files.append((path, image))
        return True

    fake_cv2.imdecode = imdecode
    fake_cv2.imwrite = imwrite

    fake_pye57 = types.SimpleNamespace(E57=lambda path: FakeE57File(root_data))

    module_name = f"extract_under_test_{id(root_data)}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)

    with mock.patch.dict(
        sys.modules,
        {
            "numpy": fake_numpy,
            "cv2": fake_cv2,
            "pye57": fake_pye57,
        },
    ):
        spec.loader.exec_module(module)

    return module, fake_cv2


class ExtractTests(unittest.TestCase):
    def test_returns_early_when_no_images_exist(self):
        module, fake_cv2 = load_extract_module({"images2D": []})

        with tempfile.TemporaryDirectory() as tmpdir:
            module.extract_and_save_images_and_metadata("dummy.e57", tmpdir)

            self.assertEqual(fake_cv2.written_files, [])
            self.assertFalse((Path(tmpdir) / "dummy" / "images").exists())
            self.assertFalse((Path(tmpdir) / "dummy" / "metadata").exists())

    def test_extracts_pinhole_image_and_pose_metadata(self):
        root_data = {
            "images2D": [
                {
                    "name": FakeValue("front_view"),
                    "pinholeRepresentation": {
                        "jpegImage": FakeBlob([1, 2, 3, 4]),
                    },
                    "pose": {
                        "translation": {
                            "x": FakeValue(1.0),
                            "y": FakeValue(2.0),
                            "z": FakeValue(3.0),
                        },
                        "rotation": {
                            "x": FakeValue(0.1),
                            "y": FakeValue(0.2),
                            "z": FakeValue(0.3),
                            "w": FakeValue(0.4),
                        },
                    },
                }
            ]
        }
        module, fake_cv2 = load_extract_module(root_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            module.extract_and_save_images_and_metadata("dummy.e57", tmpdir)

            image_path = Path(tmpdir) / "dummy" / "images" / "front_view.jpg"
            metadata_path = Path(tmpdir) / "dummy" / "metadata" / "front_view.json"

            self.assertEqual(
                fake_cv2.written_files,
                [(str(image_path), "decoded-image")],
            )
            self.assertTrue(metadata_path.exists())

            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["name"], "front_view")
            self.assertEqual(metadata["translation"], {"x": 1.0, "y": 2.0, "z": 3.0})
            self.assertEqual(
                metadata["rotation"],
                {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.4},
            )
            self.assertEqual(metadata["output_name"], "front_view")

    def test_uses_spherical_representation_when_pinhole_is_missing(self):
        root_data = {
            "images2D": [
                {
                    "name": FakeValue("panorama"),
                    "sphericalRepresentation": {
                        "jpegImage": FakeBlob([9, 8, 7]),
                    },
                }
            ]
        }
        module, fake_cv2 = load_extract_module(root_data, imdecode_result="sphere-image")

        with tempfile.TemporaryDirectory() as tmpdir:
            module.extract_and_save_images_and_metadata("dummy.e57", tmpdir)

            image_path = Path(tmpdir) / "dummy" / "images" / "panorama.jpg"
            metadata_path = Path(tmpdir) / "dummy" / "metadata" / "panorama.json"

            self.assertEqual(
                fake_cv2.written_files,
                [(str(image_path), "sphere-image")],
            )

            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata, {"name": "panorama", "output_name": "panorama"})

    def test_adds_suffixes_when_multiple_images_share_the_same_name(self):
        root_data = {
            "images2D": [
                {
                    "name": FakeValue("Panorama"),
                    "sphericalRepresentation": {
                        "jpegImage": FakeBlob([1]),
                    },
                },
                {
                    "name": FakeValue("Panorama"),
                    "sphericalRepresentation": {
                        "jpegImage": FakeBlob([2]),
                    },
                },
            ]
        }
        module, fake_cv2 = load_extract_module(root_data, imdecode_result="sphere-image")

        with tempfile.TemporaryDirectory() as tmpdir:
            module.extract_and_save_images_and_metadata("dummy.e57", tmpdir)

            first_image = Path(tmpdir) / "dummy" / "images" / "Panorama.jpg"
            second_image = Path(tmpdir) / "dummy" / "images" / "Panorama_002.jpg"
            first_metadata = Path(tmpdir) / "dummy" / "metadata" / "Panorama.json"
            second_metadata = Path(tmpdir) / "dummy" / "metadata" / "Panorama_002.json"

            self.assertEqual(
                fake_cv2.written_files,
                [
                    (str(first_image), "sphere-image"),
                    (str(second_image), "sphere-image"),
                ],
            )
            self.assertTrue(first_metadata.exists())
            self.assertTrue(second_metadata.exists())

            self.assertEqual(
                json.loads(first_metadata.read_text()),
                {"name": "Panorama", "output_name": "Panorama"},
            )
            self.assertEqual(
                json.loads(second_metadata.read_text()),
                {"name": "Panorama", "output_name": "Panorama_002"},
            )

    def test_uses_png_image_when_jpeg_image_is_missing(self):
        root_data = {
            "images2D": [
                {
                    "name": FakeValue("png_panorama"),
                    "sphericalRepresentation": {
                        "pngImage": FakeBlob([5, 6, 7]),
                    },
                }
            ]
        }
        module, fake_cv2 = load_extract_module(root_data, imdecode_result="png-image")

        with tempfile.TemporaryDirectory() as tmpdir:
            module.extract_and_save_images_and_metadata("dummy.e57", tmpdir)

            image_path = Path(tmpdir) / "dummy" / "images" / "png_panorama.jpg"
            self.assertEqual(fake_cv2.written_files, [(str(image_path), "png-image")])

    def test_skips_image_when_decoding_fails(self):
        root_data = {
            "images2D": [
                {
                    "name": FakeValue("broken_panorama"),
                    "sphericalRepresentation": {
                        "jpegImage": FakeBlob([1, 2, 3]),
                    },
                }
            ]
        }
        module, fake_cv2 = load_extract_module(root_data, imdecode_result=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            module.extract_and_save_images_and_metadata("dummy.e57", tmpdir)

            self.assertEqual(fake_cv2.written_files, [])
            self.assertFalse((Path(tmpdir) / "dummy" / "images" / "broken_panorama.jpg").exists())
            self.assertFalse((Path(tmpdir) / "dummy" / "metadata" / "broken_panorama.json").exists())


if __name__ == "__main__":
    unittest.main()
