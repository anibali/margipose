from abc import ABC, abstractmethod
import math
import torch

from . import homogeneous_to_cartesian
from .transformers import TransformerContext


class Transform(ABC):
    @abstractmethod
    def add_camera_transform(self, ctx: TransformerContext):
        pass

    @abstractmethod
    def add_image_transform(self, ctx: TransformerContext):
        pass

    @abstractmethod
    def add_point_transform(self, ctx: TransformerContext):
        pass


class AdjustColour(Transform):
    def __init__(self, brightness, contrast, saturation, hue):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def add_camera_transform(self, ctx):
        pass

    def add_image_transform(self, ctx):
        ctx.image_transformer.adjust_colour(
            self.brightness, self.contrast, self.saturation, self.hue)

    def add_point_transform(self, ctx):
        pass


class SquareCrop(Transform):
    def __init__(self):
        super().__init__()

    def add_camera_transform(self, ctx):
        image_width, image_height = ctx.image_transformer.output_size
        def crop_adjustment(camera):
            if image_height < image_width:
                camera.alpha_x *= image_width / image_height
            else:
                camera.alpha_y *= image_height / image_width
        ctx.camera_transformer.adjust_params(crop_adjustment)

    def add_image_transform(self, ctx):
        image_width, image_height = ctx.image_transformer.output_size
        if image_height < image_width:
            ctx.image_transformer.zoom(image_width / image_height, 1)
        else:
            ctx.image_transformer.zoom(1, image_height / image_width)

    def add_point_transform(self, ctx):
        pass


class HorizontalFlip(Transform):
    def __init__(self, flip_indices, do_flip):
        super().__init__()
        self.flip_indices = flip_indices
        self.do_flip = do_flip

    def add_camera_transform(self, ctx):
        if self.do_flip:
            image_width, _ = ctx.image_transformer.output_size
            def tweak_x_0(camera):
                camera.x_0 = image_width - camera.x_0
            ctx.camera_transformer.adjust_params(tweak_x_0)

    def add_image_transform(self, ctx):
        if self.do_flip:
            ctx.image_transformer.hflip()

    def add_point_transform(self, ctx):
        if self.do_flip:
            ctx.point_transformer.affine(
                A=[[-1, 0, 0],
                   [ 0, 1, 0],
                   [ 0, 0, 1]],
            )
            ctx.point_transformer.reorder_points(self.flip_indices)


class PanImage(Transform):
    def __init__(self, dx, dy):
        super().__init__()
        self.dx = dx
        self.dy = dy

    def add_camera_transform(self, ctx):
        pass

    def add_image_transform(self, ctx):
        ctx.image_transformer.translate(self.dx, self.dy)

    def add_point_transform(self, ctx):
        camera = ctx.camera_transformer.transform(ctx.orig_camera)

        ox = -self.dx / camera.alpha_x
        oy = -self.dy / camera.alpha_y
        ctx.point_transformer.affine(
            A=[[1, 0, ox],
               [0, 1, oy],
               [0, 0,  1]],
        )


class SetCentre(Transform):
    """Centre the image, transforming points such that their 2D projection remains consistent."""

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def add_camera_transform(self, ctx):
        image_width, image_height = ctx.image_transformer.output_size

        def recentre_camera(camera):
            camera.x_0 = image_width / 2
            camera.y_0 = image_height / 2
        ctx.camera_transformer.adjust_params(recentre_camera)

    def add_image_transform(self, ctx):
        image_width, image_height = ctx.image_transformer.output_size
        ctx.image_transformer.translate(image_width / 2 - self.x,
                                        image_height / 2 - self.y)

    def add_point_transform(self, ctx):
        camera = ctx.camera_transformer.transform(ctx.orig_camera)

        # NOTE: This actually warps the skeleton, bone lengths **will change**
        ox = (camera.x_0 - self.x) / camera.alpha_x
        oy = (camera.y_0 - self.y) / camera.alpha_y
        ctx.point_transformer.affine(
            A=[[1, 0, ox],
               [0, 1, oy],
               [0, 0, 1]]
        )


class SetCentreWithSimilarity(Transform):
    """Centre the image, using only similarity transformations on points."""

    def __init__(self, x, y, z_ref):
        super().__init__()
        self.x = x
        self.y = y
        self.z_ref = z_ref

    def add_camera_transform(self, ctx):
        image_width, image_height = ctx.image_transformer.output_size

        def recentre_camera(camera):
            camera.x_0 = image_width / 2
            camera.y_0 = image_height / 2
        ctx.camera_transformer.adjust_params(recentre_camera)

    def add_image_transform(self, ctx):
        image_width, image_height = ctx.image_transformer.output_size
        ctx.image_transformer.translate(image_width / 2 - self.x,
                                        image_height / 2 - self.y)

    def add_point_transform(self, ctx):
        camera = ctx.camera_transformer.transform(ctx.orig_camera)

        rays = camera.back_project(torch.DoubleTensor([self.x, self.y, 1]))
        xyzw = torch.cat([rays.narrow(-1, 0, 3), rays.narrow(-1, 2, 1) / self.z_ref], -1)
        centre3d = homogeneous_to_cartesian(xyzw)

        # Zero-centre
        ctx.point_transformer.affine(t=-centre3d)

        # Translating a 3D object in the X or Y direction will affect its 2D projection.
        # We can *approximately* correct for this by rotating the skeleton. Although
        # the 2D projection won't be exactly the same, this approach has the advantage
        # of preserving skeleton bone lengths.
        theta_x = -math.atan2(centre3d[0], self.z_ref)
        theta_y = math.atan2(centre3d[1], self.z_ref)
        y_dir = ctx.point_transformer.transform(torch.DoubleTensor([[0, 1, 0, 0]]))[0]
        ctx.point_transformer.rotate(y_dir, theta_x)
        x_dir = ctx.point_transformer.transform(torch.DoubleTensor([[1, 0, 0, 0]]))[0]
        ctx.point_transformer.rotate(x_dir, theta_y)

        # Restore depth
        ctx.point_transformer.affine(t=[0, 0, self.z_ref])


class ZoomImage(Transform):
    def __init__(self, sx, sy=None):
        super().__init__()
        self.sx = sx
        self.sy = self.sx if sy is None else sy

    def add_camera_transform(self, ctx):
        def zoom_camera(camera):
            camera.alpha_x /= self.sx
            camera.alpha_y /= self.sy
        ctx.camera_transformer.adjust_params(zoom_camera)

    def add_image_transform(self, ctx):
        ctx.image_transformer.zoom(1 / self.sx, 1 / self.sy)

    def add_point_transform(self, ctx):
        w, h = ctx.image_transformer.output_size
        camera = ctx.camera_transformer.transform(ctx.orig_camera)

        ox = (-(camera.x_0 - w / 2) * (self.sx - 1)) / camera.alpha_x
        oy = (-(camera.y_0 - h / 2) * (self.sy - 1)) / camera.alpha_y

        ctx.point_transformer.affine(
            A=[[1, 0, ox],
               [0, 1, oy],
               [0, 0,  1]]
        )


class RotateImage(Transform):
    def __init__(self, degrees):
        super().__init__()
        self.radians = math.radians(degrees)

    def add_camera_transform(self, ctx):
        pass

    def add_image_transform(self, ctx):
        ctx.image_transformer.rotate(self.radians)

    def add_point_transform(self, ctx):
        rads = self.radians
        ctx.point_transformer.affine(
            A=[[ math.cos(rads), math.sin(rads), 0],
               [-math.sin(rads), math.cos(rads), 0],
               [              0,              0, 1]],
        )


class SetImageResolution(Transform):
    def __init__(self, out_width, out_height):
        super().__init__()
        self.out_width = out_width
        self.out_height = out_height

    def add_camera_transform(self, ctx):
        in_width, in_height = ctx.image_transformer.output_size
        sx = self.out_width / in_width
        sy = self.out_height / in_height
        ctx.camera_transformer.adjust_params(lambda camera: camera.scale_image(sx, sy))

    def add_image_transform(self, ctx):
        ctx.image_transformer.resize(self.out_width, self.out_height)

    def add_point_transform(self, ctx):
        pass
