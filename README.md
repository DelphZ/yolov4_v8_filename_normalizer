# YOLO Filename Normalizer

A small Rust CLI tool to rename and organize image/label files for YOLO-style training datasets. For YoloV4, the format can be accepted by YoloV4 Darknet, most PyTorch implemented YoloV4, and also my own YoloV4 implementation using Pytorch. For YoloV8, it can be only accepted by most PyTorch implemented models.

## Features for YoloV4 (Default setting)
Rename images in a folder to img_0001.jpg, img_0002.jpg, ... (zero-padded 4 digits) when corresponding .txt label files exist.

Rename annotation file in a folder to img_0001.txt, img_0002.txt, ... when corresponding .jpg file exist.

## Features for YoloV8
Plan and rename images and labels inside train/, valid/, test/ dataset splits following a images/ labels/ layout (YOLOv8 mode).

You only need to provide the path to the dataset root folder.
The root must contain train, valid, and test subfolders, and each subfolder must have images/ and labels/ directories — e.g. root/train/images and root/train/labels.

## Prerequisites

Rust tool installed.
Run `cargo --version` and `rustup --version` to check whether you have installed Rust.

## Run / Usage
Basic usage:

`
cargo run -- <folder1> [<folder2> ...] [mode]`

The program will treat the last command-line argument as an optional "mode token" if it matches one of the known tokens:

- `v8` or `--v8` → use YOLOv8 flow

- `v4` or `--v4` → use YOLOv4 flow

##  Example usage for YoloV4

`
cargo run -- ./yolov4_example_dataset/train ./yolov4_example_dataset/valid ./yolov4_example_dataset/test
`

The program will then modify the images and annotation files in these three folders to match with yolov4 Darknet format.

## Example for YOLOv8


`cargo run -- ./yolov8_example_dataset --v8`

The program will scan dataset_root/train, dataset_root/valid, dataset_root/test and rename inside their images/ and labels/ subfolders.



