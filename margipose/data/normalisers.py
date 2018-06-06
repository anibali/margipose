from abc import ABC, abstractmethod
import torch
from scipy import optimize
from math import sqrt

from margipose.geom.camera import CameraIntrinsics
from margipose.geom import cartesian_to_homogeneous, normalise_homogeneous


class SkeletonNormaliser(ABC):
    @abstractmethod
    def normalise_skeleton(self, denorm_skel, z_ref, intrinsics, height, width):
        """Normalise the skeleton, removing scale and z position.

        Joints within the frame should have coordinate values between -1 and 1.

        Args:
            denormalised_skel (torch.DoubleTensor): The denormalised skeleton.
            z_ref (float): The depth of the plane which will become z=0.
            intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
            height (float): The image height.
            width (float): The image width.

        Returns:
            torch.DoubleTensor: The normalised skeleton.
        """
        pass

    @abstractmethod
    def denormalise_skeleton(self, norm_skel, z_ref, intrinsics, height, width):
        """Denormalise the skeleton, adding scale and z position.

        Args:
            normalised_skel (torch.DoubleTensor): The normalised skeleton.
            z_ref (float): Depth of the root joint.
            intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
            height (float): The image height.
            width (float): The image width.

        Returns:
            torch.DoubleTensor: The denormalised skeleton.
        """
        pass

    @abstractmethod
    def infer_depth(self, norm_skel, eval_scale, intrinsics, height, width):
        """Infer the depth of the root joint.

        Args:
            norm_skel (torch.DoubleTensor): The normalised skeleton.
            eval_scale (function): A function which evaluates the scale of a denormalised skeleton.
            intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
            height (float): The image height.
            width (float): The image width.

        Returns:
            float: `z_ref`, the depth of the root joint.
        """
        pass


def _scale_from_depth(z_ref, intrinsics, height, width):
    return z_ref * 0.5 * max(
        width / intrinsics.alpha_x,
        height / intrinsics.alpha_y
    )


class SquareNormaliser(SkeletonNormaliser):
    def normalise_skeleton(self, denorm_skel, z_ref, intrinsics, height, width):
        k = _scale_from_depth(z_ref, intrinsics, height, width)
        xy_n = denorm_skel.narrow(-1, 0, 2) / k
        z_n = (denorm_skel.narrow(-1, 2, 1) - z_ref) / k
        w = denorm_skel.narrow(-1, 3, 1)
        return torch.cat([xy_n, z_n, w], -1)

    def denormalise_skeleton(self, norm_skel, z_ref, intrinsics, height, width):
        k = _scale_from_depth(z_ref, intrinsics, height, width)
        xy_c = norm_skel.narrow(-1, 0, 2) * k
        z_c = norm_skel.narrow(-1, 2, 1) * k + z_ref
        w = norm_skel.narrow(-1, 3, 1)
        return torch.cat([xy_c, z_c, w], -1)

    def infer_depth(self, norm_skel, eval_scale, intrinsics, height, width):
        # NOTE: This assumes that relative bone lengths are preserved when transforming
        # between normalised skeleton <-> universal skeleton.
        k = eval_scale(norm_skel)
        # Simplified from: z_ref = mean(k * fx / (width / 2), k * fy / (height / 2))
        z_ref = k * (intrinsics.alpha_x / width + intrinsics.alpha_y / height)
        return z_ref


class PerspectiveNormaliser(SkeletonNormaliser):
    def normalise_skeleton(self, denorm_skel, z_ref, intrinsics, height, width):
        xyzw_c = normalise_homogeneous(denorm_skel)
        # Camera space XY -> Image space XY
        x_i, y_i = intrinsics.project_cartesian(xyzw_c).split(1, -1)
        # Image space XY -> Normalised XY
        x_n = x_i / (width / 2) - 1
        y_n = y_i / (height / 2) - 1
        # Camera space Z -> Normalised Z
        k = _scale_from_depth(z_ref, intrinsics, height, width)
        z_n = (xyzw_c.narrow(-1, 2, 1) - z_ref) / k
        # Join components and return
        w = torch.ones_like(z_n)
        return torch.cat([x_n, y_n, z_n, w], -1)

    def denormalise_skeleton(self, norm_skel, z_ref, intrinsics, height, width):
        x_n, y_n, z_n, w = norm_skel.split(1, -1)
        # Normalised Z -> Camera space Z
        k = _scale_from_depth(z_ref, intrinsics, height, width)
        z_c = z_n * k + z_ref
        # Normalised XY -> Image space XY
        x_i = (x_n + 1) * (width / 2)
        y_i = (y_n + 1) * (height / 2)
        # Image space XY -> Camera space XY
        rays = intrinsics.back_project(cartesian_to_homogeneous(torch.cat([x_i, y_i], -1)))
        xyzw_c = torch.cat([rays.narrow(-1, 0, 3), rays.narrow(-1, 2, 1) / z_c], -1)
        return normalise_homogeneous(xyzw_c)

    def infer_depth(self, norm_skel, eval_scale, intrinsics, height, width):
        def f(z_ref):
            z_ref = float(z_ref)
            skel = self.denormalise_skeleton(norm_skel, z_ref, intrinsics, height, width)
            k = eval_scale(skel)
            return (k - 1.0) ** 2
        near_plane = max(intrinsics.alpha_x, intrinsics.alpha_y)
        far_plane = 10000
        z_ref = float(optimize.fminbound(f, near_plane, far_plane, maxfun=200, disp=0))
        return z_ref


def make_projection_matrix(z_ref, intrinsics, height, width):
    """Build a matrix that projects from camera space into clip space.

    Args:
        z_ref (float): The reference depth (will become z=0).
        intrinsics (CameraIntrinsics): The camera object specifying focal length and optical centre.
        height (float): The image height.
        width (float): The image width.

    Returns:
        The projection matrix.
    """

    # Set the z-size (depth) of the viewing frustum to be equal to the
    # size of the portion of the XY plane at z_ref which projects
    # onto the image.
    size = z_ref * max(width / intrinsics.alpha_x, height / intrinsics.alpha_y)

    # Set near and far planes such that:
    # a) z_ref will correspond to z=0 after normalisation
    #    $z_ref = 2fn/(f+n)$
    # b) The distance from z=-1 to z=1 (normalised) will correspond
    #    to `size` in camera space
    #    $f - n = size$
    far = 0.5 * (sqrt(z_ref ** 2 + size ** 2) + z_ref - size)
    near = 0.5 * (sqrt(z_ref ** 2 + size ** 2) + z_ref + size)

    # Construct the perspective projection matrix.
    # More details: http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
    m_proj = intrinsics.matrix.new([
        [intrinsics.alpha_x / intrinsics.x_0, 0, 0, 0],
        [0, intrinsics.alpha_y / intrinsics.y_0, 0, 0],
        [0, 0, -(far + near) / (far - near), 2 * far * near / (far - near)],
        [0, 0, 1, 0],
    ])

    return m_proj


def camera_space_to_ndc(points_cam, m_proj):
    # Camera space -> clip space
    ccs = torch.matmul(points_cam, m_proj.t())
    # Clip space -> normalised device coordinates
    w = ccs.narrow(-1, 3, 1)
    return ccs / w


def ndc_to_camera_space(points_ndc, m_proj):
    # Normalised device coordinates -> clip space
    w = m_proj[2, 3] / (points_ndc.narrow(-1, 2, 1) - m_proj[2, 2])
    ccs = points_ndc * w
    # Clip space -> camera space
    return torch.matmul(ccs, m_proj.inverse().t())


class NdcNormaliser(SkeletonNormaliser):
    def normalise_skeleton(self, denorm_skel, z_ref, intrinsics, height, width):
        m_proj = make_projection_matrix(z_ref, intrinsics, height, width)
        return camera_space_to_ndc(denorm_skel, m_proj)

    def denormalise_skeleton(self, norm_skel, z_ref, intrinsics, height, width):
        m_proj = make_projection_matrix(z_ref, intrinsics, height, width)
        return ndc_to_camera_space(norm_skel, m_proj)

    def infer_depth(self, norm_skel, eval_scale, intrinsics, height, width):
        def f(z_ref):
            z_ref = float(z_ref)
            skel = self.denormalise_skeleton(norm_skel, z_ref, intrinsics, height, width)
            k = eval_scale(skel)
            return (k - 1.0) ** 2
        z_lower = max(intrinsics.alpha_x, intrinsics.alpha_y)
        z_upper = 10000
        z_ref = float(optimize.fminbound(f, z_lower, z_upper, maxfun=200, disp=0))
        return z_ref


def build_skeleton_normaliser(coord_space) -> SkeletonNormaliser:
    if coord_space == 'square':
        return SquareNormaliser()
    if coord_space == 'xy_perspective':
        return PerspectiveNormaliser()
    if coord_space == 'ndc':
        return NdcNormaliser()
    raise Exception('unrecognised normaliser type: {}'.format(coord_space))
