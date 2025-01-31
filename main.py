from datetime import datetime
from PIL import Image
import numpy as np
import os
import shutil
import time
import torch

GENERATED_IMAGES_DIR = "images"
RECONGNIZED_IMAGES_DIR = "recognized_images"
MODEL_RUNS_DIR = "runs"
MODEL_EXP_DIR = "runs/detect/exp"
LAST_RECOGNIZED_IMAGE_INDEX_FILE = "last_recongnized_image_index.txt"

RUNS_INTERVAL_SEC = 10

IMAGES_PER_RUN = 10
IMAGE_SIZE = (512, 512)

CONF_THRESHOLD = 0.1


def init_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
    model.conf = CONF_THRESHOLD
    return model


def create_images_dirs():
    if not os.path.exists(GENERATED_IMAGES_DIR):
        os.mkdir(GENERATED_IMAGES_DIR)
    if not os.path.exists(RECONGNIZED_IMAGES_DIR):
        os.mkdir(RECONGNIZED_IMAGES_DIR)


def get_last_recognized_image_index():
    if not os.path.exists(LAST_RECOGNIZED_IMAGE_INDEX_FILE):
        with open(LAST_RECOGNIZED_IMAGE_INDEX_FILE, "w") as f:
            f.write("-1")
        return -1
    with open(LAST_RECOGNIZED_IMAGE_INDEX_FILE, "r") as f:
        try:
            return int(f.read().strip())
        except ValueError:
            return -1


def update_last_recognized_image_index(new_value):
    with open(LAST_RECOGNIZED_IMAGE_INDEX_FILE, "w") as f:
        f.write(str(new_value))


def generate_images():
    filenames = []
    for image_index in range(IMAGES_PER_RUN):
        image_array = np.random.rand(IMAGE_SIZE[0], IMAGE_SIZE[1], 3) * 255
        image = Image.fromarray(image_array.astype("uint8")).convert("RGB")
        filename = os.path.join(GENERATED_IMAGES_DIR, f"{image_index}.png")
        image.save(filename)
        filenames.append(filename)
    return filenames


def save_results(results):
    if os.path.exists(MODEL_EXP_DIR):
        shutil.rmtree(MODEL_EXP_DIR)
    results.save()


def save_recognized_image(image_index, last_recognized_image_index):
    recognized_image_index = last_recognized_image_index + 1
    shutil.copy(
        os.path.join(GENERATED_IMAGES_DIR, f"{image_index}.png"),
        os.path.join(RECONGNIZED_IMAGES_DIR, f"{recognized_image_index}.png"),
    )
    shutil.copy(
        os.path.join(MODEL_EXP_DIR, f"{image_index}.jpg"),
        os.path.join(
            RECONGNIZED_IMAGES_DIR, f"{recognized_image_index}_prediction.jpg"
        ),
    )
    return recognized_image_index


def run(model, last_recognized_image_index):
    images = generate_images()

    results = model(images)
    save_results(results)

    predictions = results.pandas().xyxy
    for image_index, prediction in enumerate(predictions):
        if not prediction.empty:
            print("INFO: Non-empty prediction!")
            last_recognized_image_index = save_recognized_image(
                image_index, last_recognized_image_index
            )
    return last_recognized_image_index


def main():
    create_images_dirs()
    last_recognized_image_index = get_last_recognized_image_index()

    model = init_model()
    print("INFO: model is ready")

    while True:
        try:
            start = datetime.now()

            print("INFO: Starting new run")
            last_recognized_image_index = run(
                model, last_recognized_image_index
            )
            update_last_recognized_image_index(last_recognized_image_index)

            elapsed = (datetime.now() - start).total_seconds()
            print(f"INFO: Run took {round(elapsed, 3)} seconds")
            print(f"INFO: That's {round(elapsed / IMAGES_PER_RUN, 3)} seconds per image ({IMAGES_PER_RUN} in this run)")

            sleep_time = RUNS_INTERVAL_SEC - elapsed
            if sleep_time > 0:
                print(f"INFO: Sleeping for {round(sleep_time, 3)} seconds")
                time.sleep(sleep_time)
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
