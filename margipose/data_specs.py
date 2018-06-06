import sys
import PIL.Image
import torchvision.transforms.functional as tr
from collections import Sequence


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

    def to_dict(self):
        return {
            'type': self.__class__.__name__,
            'height': self.height,
            'width': self.width,
            'mean': self.mean,
            'stddev': self.stddev,
        }

    @classmethod
    def from_dict(cls, d):
        keys = ['mean', 'stddev']
        kwargs = { k: d[k] for k in keys }
        return cls((d['height'], d['width']), **kwargs)


class JointsSpecs:
    # There are different ways of normalising coordinates.
    # * square: Axes are aligned with camera space axes.
    # * xy_perspective: Z-axis is aligned with camera space Z-axis. XY-coords
    #                   are perspective corrected (consistent in image space).
    # * ndc: Normalised device coordinates. XY are like "xy_perspective", Z is
    #        is like "square".
    _Valid_Coord_Spaces = ['square', 'xy_perspective', 'ndc']

    def __init__(self, skeleton_desc, n_dims=3, coord_space='square'):
        assert coord_space in self._Valid_Coord_Spaces, 'invalid coord_space'
        self.skeleton_desc = skeleton_desc
        self.n_dims = n_dims
        self.coord_space = coord_space

    def to_dict(self):
        return {
            'type': self.__class__.__name__,
            'skeleton_desc': self.skeleton_desc.to_dict(),
            'n_dims': self.n_dims,
            'coord_space': self.coord_space,
        }

    @classmethod
    def from_dict(cls, d):
        from margipose.data.skeleton import SkeletonDesc
        n_dims = d['n_dims']
        coord_space = d['coord_space']
        return cls(SkeletonDesc.from_dict(d['skeleton_desc']),
                   n_dims=n_dims, coord_space=coord_space)


def _deserialise(d):
    return getattr(sys.modules[__name__], d['type']).from_dict(d)


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

    def to_dict(self):
        return {
            'input': self._input_specs.to_dict(),
            'output': self._output_specs.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(_deserialise(d['input']), _deserialise(d['output']))
