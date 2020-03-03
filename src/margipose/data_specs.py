import PIL.Image
import torchvision.transforms.functional as tr
from collections.abc import Sequence


def normalize_pixels(tensor, mean, std):
    if mean is not None:
        for t, m in zip(tensor, mean):
            t.sub_(m)
    if std is not None:
        for t, s in zip(tensor, std):
            t.div_(s)
    return tensor


def denormalize_pixels(tensor, mean, std):
    if std is not None:
        for t, s in zip(tensor, std):
            t.mul_(s)
    if mean is not None:
        for t, m in zip(tensor, mean):
            t.add_(m)
    return tensor


class ImageSpecs:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STDDEV = [0.229, 0.224, 0.225]

    def __init__(self, resolution, mean=None, stddev=None):
        if isinstance(resolution, Sequence):
            self.height, self.width = resolution
        else:
            self.height = self.width = resolution
        self.mean = mean
        self.stddev = stddev

    def convert(self, img: PIL.Image.Image):
        return normalize_pixels(tr.to_tensor(img), self.mean, self.stddev)

    def unconvert(self, tensor):
        return tr.to_pil_image(denormalize_pixels(tensor.clone(), self.mean, self.stddev), 'RGB')


class JointsSpecs:
    def __init__(self, skeleton_desc, n_dims=3):
        self.skeleton_desc = skeleton_desc
        self.n_dims = n_dims


class DataSpecs:
    """Specifications for the input and output data of a pose estimation model."""

    def __init__(self, input_specs, output_specs):
        self._input_specs = input_specs
        self._output_specs = output_specs

    @property
    def input_specs(self):
        return self._input_specs

    @property
    def output_specs(self):
        return self._output_specs
