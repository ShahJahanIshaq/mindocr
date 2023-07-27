import json
from pathlib import Path

from shapely.geometry import Polygon


class SVRD_Converter:
    """
    Format SVRD dataset into standard form.
    """

    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "det":
            self._format_det_label(Path(image_dir), label_path, output_path)
        else:
            raise ValueError("The SVRD dataset currently supports only detection.")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(label_path, "r") as f:
            label_data = json.load(f)

        with open(output_path, "w", encoding="utf-8") as out_file:
            images = sorted(image_dir.iterdir(), key=lambda path: int(path.stem))  # sort by image id

            for img_path in images:
                label = []

                img_name = img_path.name
                for anno_dict in label_data[img_name]["KV-Pairs"]:
                    anno_list = anno_dict["Values"]
                    for anno in anno_list:
                        transcription = anno["Content"]
                        polygon = anno["Coord"]
                        points = [[int(polygon[i]), int(polygon[i + 1])] for i in range(0, 8, 2)]

                        if not Polygon(points).is_valid:
                            print(f"Warning {img_path.name}: skipping invalid polygon {points}")
                            continue

                        label.append(
                            {
                                "transcription": transcription,
                                "points": points,
                            }
                        )
                for key in label_data[img_name]["Keys"].keys():
                    anno_list = label_data[img_name]["Keys"][key]
                    for anno in anno_list:
                        transcription = anno["Content"]
                        polygon = anno["Coords"]
                        points = [[int(polygon[i]), int(polygon[i + 1])] for i in range(0, 8, 2)]

                        if not Polygon(points).is_valid:
                            print(f"Warning {img_path.name}: skipping invalid polygon {points}")
                            continue

                        label.append(
                            {
                                "transcription": transcription,
                                "points": points,
                            }
                        )
                for anno in label_data[img_name]["Backgrounds"]:
                    transcription = anno["Content"]
                    polygon = anno["Coords"]
                    points = [[int(polygon[i]), int(polygon[i + 1])] for i in range(0, 8, 2)]

                    if not Polygon(points).is_valid:
                        print(f"Warning {img_path.name}: skipping invalid polygon {points}")
                        continue

                    label.append(
                        {
                            "transcription": transcription,
                            "points": points,
                        }
                    )
                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
