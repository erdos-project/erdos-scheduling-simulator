"""A helper script that retrieves the models from the Tensorflow Model zoo and saves
them in the defined directory."""

import os
import sys
import tarfile
import urllib.request

from absl import app, flags
from bs4 import BeautifulSoup, SoupStrainer

FLAGS = flags.FLAGS

# Define the flags.
flags.DEFINE_string(
    "zoo_link",
    "https://github.com/tensorflow/models/blob/master/"
    "research/object_detection/g3doc/tf2_detection_zoo.md",
    "The link where the models are hosted.",
)
flags.DEFINE_string(
    "output_dir", ".", "The directory where the results should be output."
)


def main(args):
    """Main function that retrieves the links from the model zoo."""
    if not os.path.isdir(FLAGS.output_dir):
        print(f"[x] The directory {FLAGS.output_dir} is not valid.")
        sys.exit(1)

    response = urllib.request.urlopen(FLAGS.zoo_link)
    if response.status == 200:
        model_links = {}
        for link in BeautifulSoup(
            response, "html.parser", parse_only=SoupStrainer("a")
        ):
            if link.has_attr("href") and link["href"].endswith("tar.gz"):
                model_name = link["href"].rsplit("/", 1)[-1][:-7]
                model_links[model_name] = link["href"]

        print(f"[x] Retrieved {len(model_links)} models from the link.")

        for index, model_name in enumerate(model_links.keys(), start=1):
            # Check if the model needs to be downloaded.
            output_path = os.path.join(FLAGS.output_dir, model_name + ".tar.gz")
            if os.path.exists(output_path):
                download = False
            else:
                download = True

            # Check if the model needs to be untarred.
            output_dir = os.path.join(FLAGS.output_dir, model_name)
            if os.path.exists(output_dir) and os.path.isdir(output_dir):
                untar = False
            else:
                untar = True

            # Download and untar the model.
            if untar:
                if download:
                    print(
                        f"[x] ({index}/{len(model_links)}) Retrieving {model_name} to "
                        f"{output_path}."
                    )
                    urllib.request.urlretrieve(
                        model_links[model_name], filename=output_path
                    )
                else:
                    print(
                        f"[x] ({index}/{len(model_links)}) Skipping retrieving "
                        f"{model_name} because it already exists."
                    )
                print(
                    f"[x] ({index}/{len(model_links)}) Decompressing {output_path} "
                    f"to {FLAGS.output_dir}."
                )
                model_file = tarfile.open(output_path, "r:gz")
                model_file.extractall(path=FLAGS.output_dir)
                model_file.close()
                os.remove(output_path)
            else:
                print(f"[x] ({index}/{len(model_links)}) {model_name} already exists.")
    else:
        print(f"Received {response.status} when querying {FLAGS.zoo_link}.")
        sys.exit(response.status)


if __name__ == "__main__":
    app.run(main)
