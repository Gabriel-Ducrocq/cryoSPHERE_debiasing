import torch
import einops
import numpy as np
from time import time
import matplotlib.pyplot as plt


def primal_to_fourier2d(images):
    """
    Computes the fourier transform of the images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    r = torch.fft.ifftshift(images, dim=(-2, -1))
    fourier_images = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])), dim=(-2, -1))
    return fourier_images

def fourier2d_to_primal(fourier_images):
    """
    Computes the inverse fourier transform
    fourier_images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    f = torch.fft.ifftshift(fourier_images, dim=(-2, -1))
    r = torch.fft.fftshift(torch.fft.ifft2(f, dim=(-2, -1), s=(f.shape[-2], f.shape[-1])),dim=(-2, -1)).real
    return r

def project_non_diagonal_gaussian(Gauss_mean_2d, Gauss_precisions_2d, Gauss_amplitudes, grid):
    """
    Projects a volume represented by a GMM with non diagonal covariances by integrating along the z axis. Note that in
    the case of non diagonal covariance, we cannot use the trick of computing for each axis separately and then taking
    the outer product of the two computed axis.
    Gauss_mean_2d: torch.tensor(batch_size, N_atoms, 2) of structures.
    Gauss_precisions_2d: torch.tensor(batch_size, N_atoms, 2, 2) of precision matrices for the Gaussian kernel.
    Gauss_covariances_2d: torch.tensor(batch_size, N_atoms, 2, 2) of covariances matrices for the Gaussian kernel.
    Gauss_amplitudes: torch.tensor(N_atoms, 1) of coefficients used to scale the Gausian kernels.
    grid: grid object
    return images: torch.tensor(batch_size, N_pix, N_pix)
    """
    batch_size = Gauss_mean_2d.shape[0]
    plane_coord = grid.plane_coords
    #diff is a tensor of shape (batch_size, N_residues, N_pix**2, 2)
    diff = (Gauss_mean_2d[:, :, None, :] - plane_coord[None, None, :, :])
    first_product = torch.einsum("bapk, bakl ->bapl", diff, Gauss_precisions_2d[:, :, :, :])
    #exponential is a tensor of size (batch_size, N_residues, N_pix**2)
    exponential = torch.exp(-0.5*torch.einsum("bapl, bapl -> bap", first_product, diff))*Gauss_amplitudes[None, :, :]
    images = torch.einsum("bap -> bp", exponential)
    return images.reshape(batch_size, grid.side_n_pixels, grid.side_n_pixels)


def project(Gauss_mean, Gauss_sigmas, Gauss_amplitudes, grid):
    """
    Project a volumes represented by a GMM into a 2D images, by integrating along the z axis
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3) of structures.
    Gauss_sigmas: torch.tensor(N_atoms, 1) of std for the Gaussian kernel.
    Gauss_amplitudes: torch.tensor(N_atoms, 1) of coefficients used to scale the Gausian kernels.
    grid: grid object
    return images: torch.tensor(batch_size, N_pix, N_pix)
    """
    sigmas = 2*Gauss_sigmas**2
    sqrt_amp = torch.sqrt(Gauss_amplitudes)
    proj_x = torch.exp(-(Gauss_mean[:, :, None, 0] - grid.line_coords[None, None, :])**2/sigmas[None, :, None,  0])*sqrt_amp[None, :, :]
    proj_y = torch.exp(-(Gauss_mean[:, :, None, 1] - grid.line_coords[None, None, :])**2/sigmas[None, :, None, 0])*sqrt_amp[None, :, :]
    images = torch.einsum("b a p, b a q -> b q p", proj_x, proj_y)
    return images

def structure_to_volume(Gauss_means, Gauss_sigmas, Gauss_amplitudes, grid, device):
    """
    Turn a structure into a volume using the GMM representation.
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    Gauss_sigmas: torch.tensor(N_atoms, 1)
    Gauss_amplitudes: torch.tensor(N_atoms, 1)
    grid: torch.tensor(N_pix,) where N_pix is the number of pixels on one side of the image
    return images: torch.tensor(batch_size, N_pix, N_pix, N_pix)
    """
    batch_size = Gauss_means.shape[0]
    N_pix = torch.pow(grid.line_coords.shape[0], torch.ones(1, device=device)*1/3)
    cubic_root_amp = torch.pow(Gauss_amplitudes, torch.ones(1, device=device)*1/3)
    sigmas = 2*Gauss_sigmas**2
    proj_x = torch.exp(-(Gauss_means[:, :, None, 0] - grid.line_coords[None, None, :])**2/sigmas[None, :, None, 0])*cubic_root_amp[None, :, :]
    proj_y = torch.exp(-(Gauss_means[:, :, None, 1] - grid.line_coords[None, None, :])**2/sigmas[None, :, None, 0])*cubic_root_amp[None, :, :]
    proj_z = torch.exp(-(Gauss_means[:, :, None, 2] - grid.line_coords[None, None, :])**2/sigmas[None, :, None, 0])*cubic_root_amp[None, :, :]
    volumes = torch.einsum("b a p, b a q, b a r -> b p q r", proj_x, proj_y, proj_z)    
    return volumes


def rotate_structure(Gauss_mean, precision_matrices, rotation_matrices):
    """
    Rotate a structure to obtain a posed structure.
    Gauss_mean: torch.tensor(batch_size, N_residues, 3) of atom positions
    precision_matrices: torch.tensor(batch_size, N_residues, 3, 3) of precision matrices for the GMM representation of the protein.
    rotation_matrices: torch.tensor(batch_size, 3, 3) of rotation_matrices
    return rotated_Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    """
    rotated_Gauss_mean = torch.einsum("b l k, b a k -> b a l", rotation_matrices, Gauss_mean)
    left_mult = torch.einsum("b l k, b r k m -> brlm", rotation_matrices, precision_matrices)
    rotated_precisions = torch.einsum("brij, brlj -> bril", left_mult, precision_matrices)
    return rotated_Gauss_mean, rotated_precisions


def translate_structure(Gauss_mean, translation_vectors):
    """
    Translate a structure to obtain a posed structure.
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3) of atom positions
    translation_vectors: torch.tensor(batch_size, 3) of rotation_matrices
    return translated_Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    """
    translated_Gauss_mean = Gauss_mean + translation_vectors[:, None, :]
    return translated_Gauss_mean


def apply_ctf(images, ctf, indexes):
    """
    Apply ctf to images. We multiply by -1, because we currently are white on black, but the images are more generally black on white.
    images: torch.tensor(batch_size, N_pix, N_pix)
    ctf: CTF object
    indexes: torch.tensor(batch_size, type=int), indexes of the images, to compute the ctf.
    return torch.tensor(N_batch, N_pix, N_pix) of ctf corrupted images
    """
    fourier_images = primal_to_fourier2d(images)
    fourier_images *= -ctf.compute_ctf(indexes)
    ctf_corrupted = fourier2d_to_primal(fourier_images)
    return ctf_corrupted

@torch.no_grad()
def get_radius(cov2d):
    """
    Computes the eigenvalues of 2d symmetric positive definite matrices to obtain the radius of the ellipsis.
    cov2d: torch.tensor(batch_size, N_residues, 2, 2) matrices
    """
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad
def get_rectangle(center, radius, edge_coordinate):
    """
    Draws an axis aligned square around the ellipse
    center: (batch, size, N_residues, 2) center of the Gaussian
    radius: (batch_size, N_residues, 1) radius of circle
    edge_coordinate: float, to what extent the grid goes (e.g -A, + A on both width and height)
    return: torch.tensor(batch_size, N_residues, 2) tensor with the lower bounds of the rectangle along each axis.
            torch.tensor(batch_size, N_residues, 2) tensor with the upper bounds of the rectangle along each axis.
    """
    rect_min = center - radius[:, None]
    rect_max = center + radius[:, None]
    rect_min[..., 0] = rect_min[..., 0].clip(-edge_coordinate, edge_coordinate)
    rect_min[..., 1] = rect_min[..., 1].clip(-edge_coordinate, edge_coordinate)
    rect_max[..., 0] = rect_max[..., 0].clip(-edge_coordinate, edge_coordinate)
    rect_max[..., 1] = rect_max[..., 1].clip(-edge_coordinate, edge_coordinate)
    return rect_min, rect_max


@torch.no_grad
def get_Gaussians_in_tile(rect, tile_size, grid):
    """
    Get the list of Gaussians in each tile
    rect: torch.tensor(2, batch_size, N_residues, 2) tensor of lower and uppr bounds of each rectangle along each axis
    tile_size: integer, size, in pixels, of the tiles of the tiles along each axis
    grid: grid object containing the number of pixels, their size etc...
    """
    for h in range(0, grid.side_n_pixels, tile_size):
        for w in range(0, grid.side_n_pixels, tile_size):
            #For convenient, we change the coordinates from the EMAN2 grid coordinate to (0, Extent)
            rect += grid.side_n_pixels
            #
            over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
            over_br = rect[1][..., 0].clip(max=w + tile_size - 1), rect[1][..., 1].clip(max=h + tile_size- 1)
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])  # 3D gaussian in the tile





        



 