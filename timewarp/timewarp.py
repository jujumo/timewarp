#!/usr/bin/env python3
import argparse
import logging
from PIL import Image
from typing import Optional, Tuple
import numpy as np
import os
import os.path as path
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(name)s::%(levelname)-8s: %(message)s')


def load_map(
        map_filepath: str,
        size: Optional[Tuple[int, int]] = None):
    map_image = Image.open(map_filepath)
    max_val = pow(2, 16)
    if map_image.mode != 'I':
        logger.warning('expect a 16bit gray image as map')
        # todo : max_val
        map_image = map_image.convert('I')

    if size:
        map_image = map_image.resize(size)

    map_array = np.array(map_image)
    map_array = map_array.astype(float) / max_val
    return map_array


class VerbosityParsor(argparse.Action):
    """ accept debug, info, ... or theirs corresponding integer value formatted as string."""

    def __call__(self, parser, namespace, values, option_string=None):
        assert isinstance(values, str)
        try:  # in case it represent an int, directly get it
            values = int(values)
        except ValueError:  # else ask logging to sort it out
            values = logging.getLevelName(values.upper())
        setattr(namespace, self.dest, values)


def main():
    try:
        parser = argparse.ArgumentParser(description='Description of the program.')
        parser_verbosity = parser.add_mutually_exclusive_group()
        parser_verbosity.add_argument(
            '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO, action=VerbosityParsor,
            help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
        parser_verbosity.add_argument(
            '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
        parser.add_argument('-i', '--input', required=True, help='input image directory')
        parser.add_argument('-m', '--map', required=True, help='input map file')
        parser.add_argument('-o', '--output', required=True, help='input image file')

        args = parser.parse_args()
        logger.setLevel(args.verbose)

        image_file_paths = (path.join(dp, fn) for dp, _, fs in os.walk(args.input) for fn in fs)
        image_file_paths = (image_file_path for image_file_path in image_file_paths
                            if path.splitext(image_file_path)[1] in {'.jpeg', '.jpg', '.png'})
        image_file_paths = sorted(image_file_paths)
        nb_images = len(image_file_paths)
        if not image_file_paths:
            logger.critical(f'no image found in {path.basename(args.input)}')
            raise ValueError('no image found')
        logger.info(f'found {nb_images} files.')

        im1 = Image.open(image_file_paths[0])
        canvas = np.zeros_like(im1)
        width, height = im1.size
        map_array_normed = load_map(args.map, (width, height))

        images_t = np.linspace(0, 1, num=nb_images, endpoint=True)
        map_array_idx = np.interp(map_array_normed, xp=images_t, fp=range(nb_images))

        for i, image_path in tqdm(enumerate(image_file_paths), total=nb_images):
            distance = np.absolute(map_array_idx - i)
            mask = distance <= 1.0
            if not np.any(mask):
                continue
            coefs = np.zeros_like(distance)
            coefs[mask] = 1. - distance[mask]
            image = np.array(Image.open(image_path))
            coefs = np.expand_dims(coefs, axis=2)
            canvas[mask, :] += (coefs[mask, :] * image[mask, :]).astype(np.uint8)

            if args.verbose <= logging.DEBUG:
                debug_dir_path = path.splitext(args.output)[0]
                os.makedirs(debug_dir_path, exist_ok=True)
                debug_image_path = path.join(debug_dir_path, f'{i:03}.jpg')
                Image.fromarray(canvas).save(debug_image_path)

        Image.fromarray(canvas).save(args.output)

    except Exception as e:
        logger.critical(e)
        if args.verbose <= logging.DEBUG:
            raise


if __name__ == '__main__':
    main()


