import json
from pathlib import Path

from mindocr.data.utils.polygon_utils import sort_clockwise


class GNHK_Converter:
    """
    Format GNHK dataset into standard format
    """

    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "det":
            self._format_det_label(Path(image_dir), label_path, output_path)
        else:
            raise ValueError("The GNHK dataset currently supports only detection.")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            images = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem.split("_")[-1]))  # sort by image id
            for img_path in images:
                with open(label_path / (img_path.stem + ".json"), "r") as f:
                    image_info = json.load(f)

                label = []
                for anno in image_info:
                    transcription = anno["text"]
                    if (
                        transcription == "%math%" or transcription == "%SC%" or transcription == "%NA%"
                    ):  # if marked illegible
                        transcription = "###"

                    polygon = anno["polygon"]
                    points = [
                        [int(polygon["x" + str(i)]), int(polygon["y" + str(i)])] for i in range(4)
                    ]  # reshape points (4, 2)

                    points = sort_clockwise(points).tolist()

                    label.append(
                        {
                            "transcription": transcription,
                            "points": points,
                        }
                    )
                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
