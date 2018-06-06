import unittest
from unittest.mock import patch
from ..common import TestCase

import torch
from PIL import Image

from margipose.geom.transformers import TransformerContext
from margipose.geom.transforms import HorizontalFlip


class TestHorizontalFlip(TestCase):
    def test_point_transform(self):
        ctx = TransformerContext(None, 0, 0)
        ctx.add(HorizontalFlip([], True), camera=False, image=False)

        expected = torch.Tensor([
            [-1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1],
        ])
        self.assertEqual(expected, ctx.point_transformer.matrix)

    def test_image_transform(self):
        image = Image.new('RGB', (32, 32))
        ctx = TransformerContext(None, image.width, image.height, msaa=1, border_mode='zero')
        ctx.add(HorizontalFlip([], True), camera=False, points=False)

        with patch.object(Image.Image, 'transform', return_value=None) as mock_method:
            ctx.transform(image=image)
            mock_method.assert_called_once_with(
                (32, 32),
                Image.AFFINE,
                (-1, 0, 32, 0, 1, 0),
                Image.BILINEAR
            )


if __name__ == '__main__':
    unittest.main()
