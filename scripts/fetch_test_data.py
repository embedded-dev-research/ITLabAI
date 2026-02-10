#!/usr/bin/env python3
import argparse
import pathlib
import urllib.request


DATASETS = [
    {
        "url": "https://raw.githubusercontent.com/DeepTrackAI/MNIST_dataset/main/mnist/test/1_000008.png",
        "path": "28/test1.png",
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
        "path": "lena.jpg",
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
        "path": "image.jpg",
    },
    {
        "url": "https://raw.githubusercontent.com/bpinaya/AlexNetRT/master/data/alexnet/imagenet-labels.txt",
        "path": "imagenet-labels.txt",
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/baboon.jpg",
        "path": "Imagenet_test/tench.jpg",
    },
]


def download(url: str, dest: pathlib.Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser(description="Download sample/test data for ITLabAI")
    parser.add_argument("--dest", type=pathlib.Path, required=True, help="Destination root directory")
    args = parser.parse_args()

    for item in DATASETS:
        target = args.dest / item["path"]
        download(item["url"], target)


if __name__ == "__main__":
    main()
