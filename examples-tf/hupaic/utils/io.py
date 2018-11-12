import os
import json
from pathlib import Path


def create_model_meta(model_name, input_shape, unique_lbls, output_dir: Path):
    if not output_dir.exists():
        os.makedirs(str(output_dir))

    with open(output_dir / (model_name + ".json"), 'w') as outfile:
        meta = {"input_shape": input_shape,
                "unique_lbls": unique_lbls}
        json.dump(meta, outfile)


def get_model_meta(model_name, input_dir):
    with open(input_dir / (model_name + ".json"), 'r') as file:
        meta = json.load(file)
    return meta
