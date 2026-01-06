"""
Input/Output module for shreamy.

This module provides utilities for saving and loading simulation data
in various formats, as well as reading initial conditions from external
N-body codes.
"""

import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path


# =============================================================================
# HDF5 I/O (primary format)
# =============================================================================


def save_hdf5(
    shream: "Shream",
    filename: str,
    compression: str = "gzip",
    include_potential: bool = True,
) -> None:
    """
    Save a Shream to HDF5 format.

    This is the recommended format for saving simulations, as it
    supports efficient compression and partial reads.

    Parameters
    ----------
    shream : Shream
        The Shream object to save.
    filename : str
        Output filename (should end in .hdf5 or .h5).
    compression : str, default 'gzip'
        Compression algorithm: 'gzip', 'lzf', or None.
    include_potential : bool, default True
        Whether to serialize the potential (if possible).

    Notes
    -----
    File structure:
        /particles/
            positions   : (n_times, n_particles, 3)
            velocities  : (n_times, n_particles, 3)
            masses      : (n_particles,)
            ids         : (n_particles,)
        /times          : (n_times,)
        /metadata/
            n_particles
            n_times
            creation_date
            shreamy_version
        /potential/     : (optional serialized potential info)
    """
    raise NotImplementedError


def load_hdf5(filename: str) -> "Shream":
    """
    Load a Shream from HDF5 format.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    Shream
    """
    raise NotImplementedError


def read_hdf5_times(filename: str) -> np.ndarray:
    """
    Read only the time array from an HDF5 file without loading full data.

    Useful for large files when you need to check saved times.
    """
    raise NotImplementedError


def read_hdf5_snapshot(
    filename: str,
    t: float,
) -> "ParticleSet":
    """
    Read a single snapshot from an HDF5 file.

    Parameters
    ----------
    filename : str
        Input filename.
    t : float
        Time of snapshot to read.

    Returns
    -------
    ParticleSet
    """
    raise NotImplementedError


# =============================================================================
# NumPy format (simple, fast)
# =============================================================================


def save_numpy(
    shream: "Shream",
    filename: str,
) -> None:
    """
    Save a Shream to NumPy binary format (.npz).

    Simpler than HDF5 but less feature-rich. Good for quick saves.
    """
    raise NotImplementedError


def load_numpy(filename: str) -> "Shream":
    """
    Load a Shream from NumPy format.
    """
    raise NotImplementedError


# =============================================================================
# ASCII format (human-readable)
# =============================================================================


def save_ascii(
    shream: "Shream",
    filename: str,
    t: Optional[float] = None,
    columns: Optional[list] = None,
) -> None:
    """
    Save particle data to ASCII format.

    Parameters
    ----------
    shream : Shream
    filename : str
    t : float, optional
        Time to save. If None, saves final snapshot.
    columns : list, optional
        Columns to include. Default: ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass']
    """
    raise NotImplementedError


def load_ascii(
    filename: str,
    columns: Optional[list] = None,
) -> "ParticleSet":
    """
    Load particles from ASCII file.

    Parameters
    ----------
    filename : str
    columns : list, optional
        Column specification. Default assumes [x, y, z, vx, vy, vz, mass].

    Returns
    -------
    ParticleSet
    """
    raise NotImplementedError


# =============================================================================
# FITS format (astronomy standard)
# =============================================================================


def save_fits(
    shream: "Shream",
    filename: str,
    t: Optional[float] = None,
) -> None:
    """
    Save particle data to FITS format.

    Good for interoperability with astronomy tools.
    """
    raise NotImplementedError


def load_fits(filename: str) -> "ParticleSet":
    """
    Load particles from FITS file.
    """
    raise NotImplementedError


# =============================================================================
# Interface with external N-body codes
# =============================================================================


def read_gadget(
    filename: str,
    particle_type: int = 1,
) -> "ParticleSet":
    """
    Read particles from a Gadget snapshot file.

    Parameters
    ----------
    filename : str
        Path to Gadget file (supports format 1, 2, and 3/HDF5).
    particle_type : int, default 1
        Gadget particle type to read (0=gas, 1=halo, 2=disk, etc.).

    Returns
    -------
    ParticleSet
    """
    raise NotImplementedError


def write_gadget(
    shream: "Shream",
    filename: str,
    particle_type: int = 1,
    t: Optional[float] = None,
) -> None:
    """
    Write particles to Gadget format for use in Gadget simulations.
    """
    raise NotImplementedError


def read_tipsy(filename: str) -> "ParticleSet":
    """
    Read particles from Tipsy format.
    """
    raise NotImplementedError


def read_nemo(filename: str) -> "ParticleSet":
    """
    Read particles from NEMO snapshot format.
    """
    raise NotImplementedError


def read_agama(filename: str) -> "ParticleSet":
    """
    Read particles from Agama snapshot format.
    """
    raise NotImplementedError


# =============================================================================
# Checkpoint and Restart
# =============================================================================


class Checkpoint:
    """
    Manages checkpoint files for long simulations.

    Allows saving and restoring simulation state, enabling
    restart from interruption.

    Parameters
    ----------
    directory : str
        Directory for checkpoint files.
    interval : float, optional
        Time interval between checkpoints.
    max_checkpoints : int, default 3
        Maximum number of checkpoints to keep (older ones deleted).
    """

    def __init__(
        self,
        directory: str,
        interval: Optional[float] = None,
        max_checkpoints: int = 3,
    ):
        """Initialize checkpoint manager."""
        raise NotImplementedError

    def save(self, shream: "Shream", t: float) -> str:
        """
        Save a checkpoint.

        Returns
        -------
        str
            Path to saved checkpoint.
        """
        raise NotImplementedError

    def load_latest(self) -> "Shream":
        """
        Load the most recent checkpoint.

        Returns
        -------
        Shream
        """
        raise NotImplementedError

    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.

        Returns
        -------
        list
            List of (time, filepath) tuples.
        """
        raise NotImplementedError

    def clean(self, keep_n: int = 1) -> None:
        """
        Remove old checkpoints, keeping only the most recent.
        """
        raise NotImplementedError


# =============================================================================
# Metadata utilities
# =============================================================================


def get_file_info(filename: str) -> Dict[str, Any]:
    """
    Get metadata from a shreamy file without loading full data.

    Returns
    -------
    dict
        Dictionary with keys like 'n_particles', 'n_times', 'times',
        'file_size', 'format', etc.
    """
    raise NotImplementedError


def convert_format(
    input_file: str,
    output_file: str,
    output_format: Optional[str] = None,
) -> None:
    """
    Convert between file formats.

    Parameters
    ----------
    input_file : str
    output_file : str
    output_format : str, optional
        Output format, inferred from extension if not provided.
    """
    raise NotImplementedError


# =============================================================================
# Parameter file I/O
# =============================================================================


def save_parameters(
    params: Dict[str, Any],
    filename: str,
) -> None:
    """
    Save simulation parameters to a YAML file.

    Useful for documenting and reproducing simulations.
    """
    raise NotImplementedError


def load_parameters(filename: str) -> Dict[str, Any]:
    """
    Load simulation parameters from a YAML file.
    """
    raise NotImplementedError


def create_shream_from_params(params: Dict[str, Any]) -> "Shream":
    """
    Create a Shream from a parameter dictionary.

    This is the recommended way to set up reproducible simulations.
    """
    raise NotImplementedError
