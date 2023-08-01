import json
from pathlib import Path

from shapely.geometry import Polygon
from tqdm import tqdm


class FUNSD_Converter:
    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "det":
            self._format_det_label(Path(image_dir), label_path, output_path)
        else:
            raise ValueError("FUNSD currently supports only detection.")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            images = list(image_dir.iterdir())
            for img_path in tqdm(images, total=len(images)):
                label = []
                with open(label_path / (img_path.stem + ".json")) as json_file:
                    image_info = json.load(json_file)["form"]
                for group in image_info:
                    for anno in group["words"]:
                        transcription = anno["text"]
                        points = anno["box"]
                        points = [
                            [points[0], points[1]],
                            [points[2], points[1]],
                            [points[2], points[3]],
                            [points[0], points[3]],
                        ]

                        if not Polygon(points).is_valid:
                            print(f"Warning {img_path.name}: skipping invalid polygon {img_path}")
                            continue

                        label.append(
                            {
                                "transcription": transcription,
                                "points": points,
                            }
                        )
                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
