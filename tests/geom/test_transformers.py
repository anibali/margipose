import unittest
from ..common import TestCase

from PIL import Image
import numpy as np
import torch

from margipose.geom.transformers import ImageTransformer, PointTransformer


class TestImageTransformer(TestCase):
    def test_hflip(self):
        image = Image.fromarray(np.uint8([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]))
        trans = ImageTransformer(*image.size)
        trans.hflip()
        expected = np.uint8([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ])
        actual = np.asarray(trans.transform(image))
        self.assertEqual(expected, actual)

    def test_rotate(self):
        image = Image.fromarray(np.uint8([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]))
        trans = ImageTransformer(*image.size)
        trans.rotate(np.pi / 2)
        expected = np.uint8([
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ])
        actual = np.asarray(trans.transform(image))
        self.assertEqual(expected, actual)

    def test_zoom(self):
        image = Image.fromarray(np.uint8([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]))
        trans = ImageTransformer(*image.size)
        trans.zoom(1 / 3)
        expected = np.uint8([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        actual = np.asarray(trans.transform(image))
        self.assertEqual(expected, actual)

    def test_untransform(self):
        trans = ImageTransformer(3, 3)
        trans.hflip()
        trans.rotate(-np.pi / 2)
        trans.translate(1, 0)
        transformed_image = Image.fromarray(np.uint8([
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ]))
        expected = np.uint8([
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
        ])
        actual = np.asarray(trans.untransform(transformed_image))
        self.assertEqual(expected, actual)

class TestPointTransformer(TestCase):
    def test_is_similarity(self):
        similarities = [
            # Identity
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            # Translation
            [[1, 0, 0, 2],
             [0, 1, 0, 2],
             [0, 0, 1, 2],
             [0, 0, 0, 1]],
            # Mirroring
            [[-1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            # Rotating
            [[1, 0, 0, 0],
             [0, 0, -1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]],
            # Uniform scaling
            [[2, 0, 0, 0],
             [0, 2, 0, 0],
             [0, 0, 2, 0],
             [0, 0, 0, 1]],
            # Mix
            [[-2, 0, 0, 1],
             [0, 0, 2, 2],
             [0, -2, 0, 3],
             [0, 0, 0, 1]],
        ]
        non_similarities = [
            # Non-affine transform
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [2, 0, 0, 1]],
            # Non-uniform scaling
            [[1, 0, 0, 0],
             [0, 2, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
        ]

        for matrix in similarities:
            trans = PointTransformer()
            trans.matrix.copy_(torch.DoubleTensor(matrix))
            self.assertTrue(trans.is_similarity())

        for matrix in non_similarities:
            trans = PointTransformer()
            trans.matrix.copy_(torch.DoubleTensor(matrix))
            self.assertFalse(trans.is_similarity())


if __name__ == '__main__':
    unittest.main()
