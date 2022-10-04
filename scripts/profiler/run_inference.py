import os
from time import sleep, time

import numpy as np
import tensorflow as tf
from absl import app, flags
from object_detection.builders import model_builder
from object_detection.utils import config_util
from PIL import Image
from six import BytesIO

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "The directory where the model is stored.")
flags.register_validator(
    "model_dir", lambda val: val is not None, "Model directory should be specified."
)
flags.DEFINE_string("image_path", None, "The path where the image is stored.")
flags.register_validator(
    "image_path", lambda val: val is not None, "Image path should be specified."
)

WARMUP_SAMPLES = 5


def decode_image(image_path):
    image_data = tf.io.gfile.GFile(image_path, "rb").read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image_np = (
        np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    )
    return tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)


def main(args):
    # Sleep so that the PID can be attached to the profilers.
    sleep(10)
    print(f"[x] Initiating the main method at {time()}.")
    # Check that the model exists.
    pipeline_config = os.path.join(FLAGS.model_dir, "pipeline.config")
    model_directory = os.path.join(FLAGS.model_dir, "checkpoint")
    if not os.path.exists(pipeline_config) or not os.path.isdir(model_directory):
        print(f"[x] The model at {FLAGS.model_dir} was not found.")
        return 1

    # Set memory growth limits for all GPUs.
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Load the model.
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs["model"]
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    checkpoint = tf.compat.v2.train.Checkpoint(model=detection_model)
    checkpoint.restore(os.path.join(model_directory, "ckpt-0")).expect_partial()
    print(f"[x] The model was restored at {time()}.")

    # Decode the image.
    print(f"[x] Beginning the image decoding process at {time()}.")
    image_tensor = decode_image(FLAGS.image_path)
    print(f"[x] Finished the image decoding process at {time()}.")

    # Preprocess the image.
    print(f"[x] Beginning the image pre-processing at {time()}.")
    image, shapes = detection_model.preprocess(image_tensor)
    print(f"[x] Finished the image pre-processing at {time()}.")

    # Predict the image.
    for _ in range(WARMUP_SAMPLES):
        prediction_dict = detection_model.predict(image, shapes)
    print(f"[x] Beginning the image prediction at {time()}.")
    prediction_dict = detection_model.predict(image, shapes)
    print(f"[x] Finished the image prediction at {time()}.")

    # Postprocess the results.
    print(f"[x] Beginning the image postprocessing at {time()}.")
    detection_model.postprocess(prediction_dict, shapes)
    print(f"[x] Finished the image postprocessing at {time()}.")


if __name__ == "__main__":
    app.run(main)
