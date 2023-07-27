import csv
import json
from pathlib import Path

# import requests
from shapely.geometry import Polygon

# data_dir = Path("../data/all/tianchi")
# imgs_save_dir = data_dir / "train1"
# imgs_save_dir.mkdir(exist_ok=True)

# csv_filenames = ["train1.csv"]

# for filename in csv_filenames:
#     with open(data_dir / filename) as csvfile:
#         reader = csv.DictReader(csvfile, fieldnames=("id", "raw_data", "label"))
#         next(reader)
#         for row in tqdm(reader):
#             img_id = row["id"]
#             raw_data = json.loads(row["raw_data"])
#             img_url = raw_data["tfspath"]
#             response = requests.get(img_url)
#             if response.status_code:
#                 fp = open(imgs_save_dir / (str(img_id) + ".jpg"), 'wb')
#                 fp.write(response.content)
#                 fp.close()


class TIANCHI_Converter:
    """
    Format tianchi dataset into standard format.
    """

    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "det":
            self._format_det_label(Path(image_dir), label_path, output_path)
        else:
            raise ValueError("The tianchi dataset currently supports only detection.")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            with open(label_path) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=("id", "raw_data", "label"))
                next(reader)
                for row in reader:
                    label = []
                    img_path = row["id"] + ".jpg"
                    image_info = json.loads(row["label"])
                    for anno in image_info[0]:
                        transcription = json.loads(anno["text"])["text"]
                        polygon = anno["coord"]
                        points = [[float(polygon[i]), float(polygon[i + 1])] for i in range(0, 8, 2)]

                        if not Polygon(points).is_valid:
                            print(f"Warning {img_path.name}: skipping invalid polygon {points}")
                            continue

                        label.append(
                            {
                                "transcription": transcription,
                                "points": points,
                            }
                        )
                    # img_path = img_path.name if self._relative else str(img_path)
                    out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
