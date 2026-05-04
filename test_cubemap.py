import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent
MODULE_PATH = REPO_ROOT / "cubemap.py"


def load_cubemap_module():
    module_name = "cubemap_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeImage:
    def __init__(self, height, width, channels=3):
        self.shape = (height, width, channels)

    def __getitem__(self, key):
        rows, cols = key[:2]

        def bounds(indexer, size):
            if isinstance(indexer, slice):
                start = 0 if indexer.start is None else indexer.start
                stop = size if indexer.stop is None else indexer.stop
                return start, stop
            return indexer, indexer + 1

        row_start, row_stop = bounds(rows, self.shape[0])
        col_start, col_stop = bounds(cols, self.shape[1])
        return FakeImage(row_stop - row_start, col_stop - col_start, self.shape[2])


class CubemapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_cubemap_module()

    def test_generate_cubemap_faces_returns_standard_order(self):
        with mock.patch.object(
            self.module,
            "_render_face",
            side_effect=lambda image, face_name, face_size: f"{face_name}:{face_size}",
        ):
            faces = self.module.generate_cubemap_faces("image", 64)

        self.assertEqual(
            faces,
            [
                ("posx", "posx:64"),
                ("negx", "negx:64"),
                ("posy", "posy:64"),
                ("negy", "negy:64"),
                ("posz", "posz:64"),
                ("negz", "negz:64"),
            ],
        )

    def test_valid_panorama_uses_highest_face_resolution_and_writes_six_faces(self):
        fake_faces = [(face_name, f"{face_name}-image") for face_name in self.module.FACE_NAMES]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir) / "cubemaps"
            image_path = Path(tmpdir) / "p000016.jpg"
            image_path.write_text("placeholder")

            with (
                mock.patch.object(self.module, "load_image", return_value=object()),
                mock.patch.object(
                    self.module,
                    "normalize_to_2_to_1",
                    return_value=(FakeImage(3840, 7680), False),
                ),
                mock.patch.object(
                    self.module,
                    "generate_cubemap_faces",
                    return_value=fake_faces,
                ) as generate_faces,
                mock.patch.object(self.module, "save_face_image") as save_face_image,
            ):
                converted = self.module.convert_image_to_cubemap(image_path, output_folder, quality=92)

        self.assertTrue(converted)
        generate_faces.assert_called_once()
        self.assertEqual(generate_faces.call_args.args[1], 1920)
        self.assertEqual(save_face_image.call_count, 6)
        written_names = [call.args[0].name for call in save_face_image.call_args_list]
        self.assertEqual(written_names, [f"{face_name}.jpg" for face_name in self.module.FACE_NAMES])
        written_parent = save_face_image.call_args_list[0].args[0].parent.name
        self.assertEqual(written_parent, "p000016")

    def test_non_2_to_1_image_is_center_cropped_for_best_effort_conversion(self):
        normalized, best_effort = self.module.normalize_to_2_to_1(FakeImage(3000, 5000))

        self.assertTrue(best_effort)
        self.assertEqual(normalized.shape[:2], (2500, 5000))

    def test_default_output_folder_is_nested_under_images_folder(self):
        output_folder = self.module.default_cubemap_output_folder("ext/Scans/images")

        self.assertEqual(Path(output_folder), Path("ext/Scans/images/cubemaps"))

    def test_unreadable_images_are_skipped_without_aborting_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_folder = Path(tmpdir) / "images"
            input_folder.mkdir()
            (input_folder / "broken.jpg").write_text("placeholder")

            with mock.patch.object(self.module, "load_image", return_value=None):
                result = self.module.convert_folder(input_folder)

        self.assertEqual(result, {"processed": 0, "skipped": 1})


if __name__ == "__main__":
    unittest.main()
