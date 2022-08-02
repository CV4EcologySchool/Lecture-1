#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The lecture materials for Lecture 1: Dataset Prototyping and Visualization
"""
import logging
import pprint
from logging.handlers import TimedRotatingFileHandler
from os.path import abspath

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, io, transforms, utils


DAYS = 21

log = None


def _setup_logging():
    """
    Setup Python's built in logging functionality with on-disk logging, and prettier logging with Rich
    """
    # Import Rich
    import rich
    from rich.logging import RichHandler
    from rich.style import Style
    from rich.theme import Theme

    name = 'lecture_1'

    # Setup placeholder for logging handlers
    handlers = []

    # Configuration arguments for console, handlers, and logging
    console_kwargs = {
        'theme': Theme(
            {
                'logging.keyword': Style(bold=True, color='yellow'),
                'logging.level.notset': Style(dim=True),
                'logging.level.debug': Style(color='cyan'),
                'logging.level.info': Style(color='green'),
                'logging.level.warning': Style(color='yellow'),
                'logging.level.error': Style(color='red', bold=True),
                'logging.level.critical': Style(color='red', bold=True, reverse=True),
                'log.time': Style(color='white'),
            }
        )
    }
    handler_kwargs = {
        'rich_tracebacks': True,
        'tracebacks_show_locals': True,
    }
    logging_kwargs = {
        'level': logging.INFO,
        'format': '[%(name)s] %(message)s',
        'datefmt': '[%X]',
    }

    # Add file-baesd log handler
    handlers.append(
        TimedRotatingFileHandler(
            filename=f'{name}.log',
            when='midnight',
            backupCount=DAYS,
        ),
    )

    # Add rich (fancy logging) log handler
    rich.reconfigure(**console_kwargs)
    handlers.append(RichHandler(**handler_kwargs))

    # Setup global logger with the handlers and set the default level to INFO
    logging.basicConfig(handlers=handlers, **logging_kwargs)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log = logging.getLogger(name)

    return log


def load_mnist():
    """
    Load the MNIST dataset from PyTorch (download if needed) and return a DataLoader

    MNIST is a sample dataset for machine learning, each image is 28-pixels high and 28-pixels wide (1 color channel)
    """
    root = abspath('datasets')

    # Load the training data from the MNIST dataset
    dataset = datasets.MNIST(
        root, train=True, transform=transforms.ToTensor(), download=True
    )

    # Create a DataLoader that takes the dataset and allows a loop to iterate over 4-D tensors
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=1
    )

    return dataset, dataloader


def load_image_folder_dataset():
    """
    Load a folder of images of zebras, where the folder names are the IDs of the animals
    """
    root = abspath('zebras')

    transform = transforms.Compose(
        [
            transforms.Resize(
                [224, 224]
            ),  # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=1
    )

    return dataset, dataloader


def print_data_stats(dataset, index=0):
    """
    Print out the dataset and show an example image in the terminal
    """
    log.info(dataset)

    log.info(f'Dataset shape: {dataset.data.shape}')

    log.info(f'Dataset classes:\n{pprint.pformat(dataset.classes)}')

    # Load example image (as Tensor)
    image = dataset.data[index]
    log.info(f'Tensor image shape: {image.shape}')
    log.info(f'Tensor image dtype: {image.dtype}')
    log.info(f'Tensor image min: {torch.min(image)}')
    log.info(f'Tensor image max: {torch.max(image)}')

    # Convert to Numpy array
    image = image.numpy()
    log.info(f'Numpy image shape: {image.shape}')
    log.info(f'Numpy image dtype: {image.dtype}')
    log.info(f'Numpy image min: {np.min(image)}')
    log.info(f'Numpy image max: {np.max(image)}')
    log.info(f'Numpy image mean: {np.mean(image)}')

    # Print image out as ASCII text
    mean = np.mean(image)
    width, height = image.shape
    log.info('-' * (width + 2))
    for column in range(width):
        line = '|'
        for row in range(height):
            line += 'X' if image[column][row] >= mean else ' '
        line += '|'
        log.info(line)
    log.info('-' * (width + 2))
    log.info(f'GROUND-TRUTH LABEL: {dataset.targets[index]}')
    log.info('-' * (width + 2))

    # Get all of the target labels and convert from tensor to Numpy array
    targets = dataset.targets.numpy()

    # Print the distributions of labels in the MNIST train set
    bins = list(range(11))
    totals, labels = np.histogram(targets, bins=bins)
    for label, total in zip(labels, totals):
        log.info(f'Label {label}: {total}')


def show_label_histogram(dataset):
    """
    Plot the distrubition of labels as a histogram with Matplotlib
    """
    targets = dataset.targets.numpy()

    # Print the distributions of labels in the MNIST train set
    labels = sorted(set(targets))
    bins = list(range(len(labels) + 1))

    # Compute and plot the histogram
    plt.hist(targets, bins=bins, histtype='bar', facecolor='b', rwidth=0.9, alpha=0.5)

    # Setup the chart's axis and labels
    plt.xticks(np.array(bins[:-1]) + 0.5, labels)
    plt.title('Distribution of Labels')
    plt.xlabel('Ground-truth Label')
    plt.ylabel('Number of examples')

    # Show the figure to the user (or embed into notebook)
    plt.show()


def write_tensor_to_disk(data, label):
    log.info('Writing data to disk:')
    log.info(f'\tshape: {data.shape}')
    log.info(f'\tdtype: {data.dtype}')
    log.info(f'\tmin:   {torch.min(data)}')
    log.info(f'\tmax:   {torch.max(data)}')
    log.info(f'\tlabel: {label}')

    # Write out tensor to disk
    # Convert range from [0.0, 1.0] to [0, 255]
    data_torch = data * 255.0
    # Round to nearest whole number
    data_torch = torch.round(data_torch)
    # Truncate negative numbers
    data_torch[data_torch < 0] = 0
    # Truncate out-of-bounds numbers
    data_torch[data_torch > 255] = 255
    # Convert from existing torch.float32 to torch.uint8 (unsigned 8-bit integer)
    data_torch = data_torch.to(dtype=torch.uint8)
    # Write image to disk
    io.write_jpeg(data_torch, f'output.tensor.label_{label}.jpg')

    # Write tensor to disk (using Numpy and OpenCV)
    # First, convert to Numpy array
    data_np = data.numpy()
    log.info('Writing data to disk:')
    log.info(f'\tshape: {data_np.shape}')
    log.info(f'\tdtype: {data_np.dtype}')
    log.info(f'\tmin:   {np.min(data_np)}')
    log.info(f'\tmax:   {np.max(data_np)}')
    log.info(f'\tlabel: {label}')

    # Transpose C,W,H to W,H,C for OpenCV
    data_np = np.transpose(data_np, (1, 2, 0))
    # Convert range from [0.0, 1.0] to [0, 255]
    data_np = data_np * 255.0
    # Round to nearest whole number
    data_np = np.around(data_np)
    # Truncate negative numbers
    data_np[data_np < 0] = 0
    # Truncate out-of-bounds numbers
    data_np[data_np > 255] = 255
    # Convert from existing np.float32 to np.uint8 (unsigned 8-bit integer)
    data_np = data_np.astype(np.uint8)
    # Invert colors from RGB to BGR (for OpenCV) if it is a 3-channel color image
    w, h, c = data_np.shape
    if c == 3:
        data_np = data_np[:, :, ::-1]
    # Write image to disk with OpenCV
    cv2.imwrite(f'output.numpy.label_{label}.jpg', data_np)

    return data_torch, data_np


def write_tensor_grid_to_disk(datas):
    log.info('Writing data grid to disk:')
    log.info(f'\tshape: {datas.shape}')
    log.info(f'\tdtype: {datas.dtype}')

    # Create grid of images using the images in the batch
    data_grid = utils.make_grid(datas)

    # Convert range from [0.0, 1.0] to [0, 255]
    data_grid = data_grid * 255.0
    # Round to nearest whole number
    data_grid = torch.round(data_grid)
    # Truncate negative numbers
    data_grid[data_grid < 0] = 0
    # Truncate out-of-bounds numbers
    data_grid[data_grid > 255] = 255
    # Convert from existing torch.float32 to torch.uint8 (unsigned 8-bit integer)
    data_grid = data_grid.to(dtype=torch.uint8)

    # Write image to disk
    io.write_jpeg(data_grid, 'output.tensor.grid.jpg')

    # Convert grid to Numpy (we can use the whitening steps from before, no need to re-do)
    data_grid = data_grid.numpy()
    # Transpose C,W,H to W,H,C for OpenCV
    data_grid = np.transpose(data_grid, (1, 2, 0))

    # Invert colors from RGB to BGR (for OpenCV) if it is a 3-channel color image
    w, h, c = data_grid.shape
    if c == 3:
        data_grid = data_grid[:, :, ::-1]

    # Write image to disk with OpenCV
    cv2.imwrite('output.numpy.grid.jpg', data_grid)

    return data_grid


def show_example_bounding_boxes(data_torch):
    # Define bounding boxes
    boxes = torch.tensor([[50, 50, 100, 100], [25, 25, 125, 125]], dtype=torch.float)
    colors = ['blue', 'yellow']

    # Draw over the tensor and return a new result tensor
    result = utils.draw_bounding_boxes(data_torch, boxes, colors=colors, width=5)

    # Convert to Numpy and display bouding boxes
    result = result.numpy()
    result = np.transpose(result, (1, 2, 0))

    plt.imshow(result)
    plt.show()


def show_example_points(data_torch):
    # Define keypoints
    keypoints = torch.tensor(
        [
            [
                [50, 50, 1.0000],
                [50, 100, 1.0000],
                [100, 50, 1.0000],
                [100, 100, 1.0000],
                [150, 150, 1.0000],
            ]
        ],
        dtype=torch.float,
    )

    # Define connected skeleton (optional)
    skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (1, 4),
    ]

    # Draw over the tensor and return a new result tensor
    result = utils.draw_keypoints(
        data_torch, keypoints, connectivity=skeleton, colors='blue', radius=4, width=3
    )

    # Convert to Numpy and display bouding boxes
    result = result.numpy()
    result = np.transpose(result, (1, 2, 0))

    plt.imshow(result)
    plt.show()


@click.command()
def lecture():
    """
    Main function for Lecture 1: Dataset Prototyping and Visualization
    """
    global log

    log = _setup_logging()

    ################################################################################
    # Load MNIST
    dataset, dataloader = load_mnist()

    # Visualize dataset
    index = 0
    print_data_stats(dataset, index)
    # print_data_stats(dataset, index=1)
    # print_data_stats(dataset, index=2)

    # Show histogram of labels with Matplotlib
    show_label_histogram(dataset)

    ################################################################################
    # Load the first batch from the dataloader
    datas, labels = next(iter(dataloader))
    log.info(f'Batch shape:  {datas.shape}')
    log.info(f'Batch labels: {labels}')

    # zip N data values with N labels (N=16 by default)
    #   creates zipped = [(data_1, label_1), (data_2, label_2), ..., (data_N, label_N)]
    #   where len(zipped) == N
    #   note: zip returns an iterator, use the list() cast to make the data usable
    batch = list(zip(datas, labels))

    # Grab the first (data, label) pair from the zipped list
    example = batch[0]
    data, label = example

    # Write the data to disk
    write_tensor_to_disk(data, label)

    # Write a grid of datas to disk
    data_grid = write_tensor_grid_to_disk(datas)

    # Show image grid with Matplotlib
    plt.imshow(data_grid)
    plt.show()

    ################################################################################
    # Load zebras from a folder of images
    dataset, dataloader = load_image_folder_dataset()
    log.info(dataset)

    # Write an example image to disk
    data, index = dataset[5]
    names = dataset.classes
    name = names[index]

    # Write the data to disk
    data_torch, data_np = write_tensor_to_disk(data, name)

    ################################################################################
    # Invert colors and Save
    data_np = data_np[:, :, ::-1]
    cv2.imwrite(f'output.numpy.label_{name}.inverted.jpg', data_np)

    ################################################################################
    # Draw example bounding boxes onto Tensor
    show_example_bounding_boxes(data_torch)

    ################################################################################
    # Draw points with connections onto Tensor
    show_example_points(data_torch)


if __name__ == '__main__':
    # Common boiler-plating needed to run the code from the command line as `python lecture.py` or `./lecture.py`
    # This if condition will be False if the file is imported
    lecture()
