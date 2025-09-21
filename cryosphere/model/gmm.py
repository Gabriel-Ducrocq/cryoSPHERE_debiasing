import roma
import torch
import numpy as np
from typing import Union
from dataclasses import dataclass



@dataclass
class Gaussian:
    mus: Union[torch.Tensor, np.ndarray]
    sigmas: Union[torch.Tensor, np.ndarray]
    amplitudes: Union[torch.Tensor, np.ndarray]

class Gaussian_splatting(torch.nn.Module):
    """
    Object containing the parameters of the GMM representation of the protein:
    - the Gaussian means
    - the initial Gaussian standard deviations
    - the initial amplitudes
    We will learn the Gaussian standard deviations and initial amplitudes as part of the Gaussian splatting process.
    """
    def __init__(self, mus, sigmas, amplitudes):
        """
        Initialize the GMM representation of the protein, definining the means, amplitudes, R and S matrices of the
        Gaussians so that Sigma = RSSR^T
        The normal matrix R is represented as a quaternion.
        mus: torch.tensor(N_residues, 3)
        sigmas: torch.tensor(N_residues, 3)
        amplitudes: torch.tensor(N_residues, 1)
        """
        self.mus = torch.nn.Parameters(data=mus, requires_grad = True)
        self.amplitudes = torch.nn.Parameters(data= amplitudes, requires_grad = True)
        self.S = torch.nn.Parameters(data= sigmas, requires_grad = True)
        self.quaternions = torch.zeros(self.mus.shape[0], 4)
        self.quaternions[:, -1] = 1
        self.quaternions = torch.nn.Parameters(data= self.quaternions, requires_grad = True)

    def compute_R(self):
        """
        Computes the R matrix for each residue
        return: torch.tensor(N_residues, 3, 3)
        """
        R = roma.unitquat_to_rotmat(self.quaternions)
        return R

    def get_S(self):
        """
        Return the diagonal elements of the S matrix
        return: torch.tensor(N_residues, 3)
        """
        return self.S

    def compute_precision_matrices(self):
        """
        Compute the precision matrices based on R an S for each residue
        return: torch.tensor(N_residues, 3, 3) precision matrices for each residue
        """
        R = self.compute_R()
        S = self.get_S()
        RS = R*S[:, None, :]
        return torch.einsum("rij, rlj -> ril", RS, RS)





class BaseGrid(torch.nn.Module):
	"""
	Grid spanning origin, to origin + (side_shape - 1) * voxel_size, for the coordinate of each pixel.
	"""
	def __init__(self, side_n_pixels, voxel_size, origin=None, device="cpu"):
		"""
		:param side_n_pixels: integer, number of pixel on each side of the image.
		:param voxel_size: float, size of each pixel in Å.
		:param origin: float, origin on the image.
		:param device: torch device on which to perform the computations.
    	"""
		super().__init__()
		self.side_n_pixels = side_n_pixels
		self.voxel_size = voxel_size
		if not origin:
			origin = 0

		self.origin = origin

		line_coords = torch.linspace(origin, (side_n_pixels - 1) * voxel_size + origin, side_n_pixels, device=device)
		self.register_buffer("line_coords", line_coords)
		[xx, yy] = torch.meshgrid([self.line_coords, self.line_coords], indexing="ij")
		plane_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
		self.register_buffer("plane_coords", plane_coords)
		self.plane_shape = (self.side_n_pixels, self.side_n_pixels)

		[xx, yy, zz] = torch.meshgrid([self.line_coords, self.line_coords, self.line_coords])
		vol_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
		self.register_buffer("vol_coords", vol_coords)
		self.vol_shape = (self.side_n_pixels, self.side_n_pixels, self.side_n_pixels)



class EMAN2Grid(BaseGrid):
	"""EMAN2 style grid for the coordinates of each pixel.
	origin set to -(side_shape // 2) * voxel_size
	"""
	def __init__(self, side_shape, voxel_size, device="cpu"):
		"""
		:param side_shape: integer, number of pixel on each side of the image.
		:param voxel_size: float, size of each pixel in Å.
		"""
		origin = -side_shape // 2 * voxel_size
		super().__init__(side_n_pixels=int(side_shape), voxel_size=voxel_size, origin=origin, device=device)




