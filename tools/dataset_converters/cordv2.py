import json
import shutil
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from mindocr.data.utils.polygon_utils import sort_clockwise


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

        if list(image_dir.glob("*.parquet")):  # if the path to the folder with .parquet files
            shutil.rmtree(image_dir / self._imgs_folder, ignore_errors=True)
            shutil.rmtree(image_dir / self._labels_folder, ignore_errors=True)
            self._make_data_from_parquet(image_dir)

            image_dir = image_dir / self._imgs_folder
            label_path = label_path / self._labels_folder

        if task == "det":
            self._format_det_label(image_dir, label_path, output_path)
        else:
            raise ValueError("The CORDv2 dataset currently supports only detection.")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            images = sorted(image_dir.iterdir(), key=lambda path: int(path.stem))  # sort by image id
            for img_path in tqdm(images):
                with open(label_path / (img_path.stem + ".json"), "r") as f:
                    image_info = json.load(f)

                labels = [
                    {"transcription": anno["text"], "points": anno["quad"]}
                    for line in image_info["valid_line"]
                    for anno in line["words"]
                ]

                # Mark separation symbols (such as '---' or '===') as ignored to avoid them being classified as
                # False Positive.
                labels.extend(
                    [
                        {"transcription": "###", "points": anno["quad"]}
                        for line in image_info["repeating_symbol"]
                        for anno in line
                    ]
                )

                labels.extend(
                    [{"transcription": "###", "points": anno} for line in image_info["dontcare"] for anno in line]
                )

                for label in labels:
                    label["points"] = [
                        [int(label["points"]["x" + str(i)]), int(label["points"]["y" + str(i)])] for i in range(1, 5)
                    ]  # reshape points (4, 2)
                    label["points"] = sort_clockwise(label["points"]).tolist()

                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(labels, ensure_ascii=False) + "\n")

    def _make_data_from_parquet(self, image_dir: Path):
        df = pd.read_parquet(image_dir, engine="auto")

        img_save_dir = image_dir / self._imgs_folder
        img_save_dir.mkdir(exist_ok=True)
        labels_save_dir = image_dir / self._labels_folder
        labels_save_dir.mkdir(exist_ok=True)

        for img_num, data in tqdm(df.iterrows(), total=len(df.index)):
            img = Image.open(BytesIO(data["image"]["bytes"]))
            img.save(img_save_dir / f"{img_num:04d}.{img.format.lower()}")

            with open(labels_save_dir / f"{img_num:04d}.json", "w") as label_file:
                label_file.write(data["ground_truth"])
