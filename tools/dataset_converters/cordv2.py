import json
import shutil
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm


class CORDV2_Converter:
    """
    Format CORDV2 dataset into standard format

    """

    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"
        self._imgs_folder = "images"
        self._labels_folder = "labels"

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        image_dir = Path(image_dir)
        try:
            shutil.rmtree(image_dir / self._imgs_folder)
            shutil.rmtree(image_dir / self._labels_folder)
        except FileNotFoundError:
            pass
        self._make_data_from_parquet(image_dir)

        if task == "det":
            self._format_det_label(image_dir / self._imgs_folder, label_path / self._labels_folder, output_path)
        else:
            raise ValueError("The CORDv2 dataset currently supports only detection.")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            images = sorted(image_dir.iterdir(), key=lambda path: int(path.stem))  # sort by image id
            for img_path in images:
                with open(label_path / (img_path.stem + ".json"), "r") as f:
                    image_info = json.load(f)

                label = []
                for line in image_info["valid_line"]:
                    for anno in line["words"]:
                        transcription = anno["text"]
                        polygon = anno["quad"]
                        points = [
                            [int(polygon["x" + str(i)]), int(polygon["y" + str(i)])] for i in range(1, 5)
                        ]  # reshape points (4, 2)

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

    def _make_data_from_parquet(self, image_dir: Path):
        df = pd.read_parquet(image_dir, engine="auto")

        # print(df['image'])

        for img_num in tqdm(range(len(df["image"]))):
            img_save_dir = image_dir / self._imgs_folder
            img_save_dir.mkdir(exist_ok=True)
            labels_save_dir = image_dir / self._labels_folder
            labels_save_dir.mkdir(exist_ok=True)

            img_file = open(img_save_dir / (str(img_num) + ".jpg"), "wb")
            img = df["image"][img_num]["bytes"]
            img_file.write(img)

            label = json.loads(df["ground_truth"][img_num])
            with open(labels_save_dir / (str(img_num) + ".json"), "w") as label_file:
                json.dump(label, label_file)
