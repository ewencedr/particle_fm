"""Module with functions related to calculating jet substructure.

All of these are essentially based around pyjet.PseudoJet objects and/or
their constituent particles and calculates some of the most popular
high-level substructure observables.

from https://github.com/DebajyotiS/PC-JeDi/blob/EPiC-JeDi/src/jet_substructure.py
"""

import os
from pathlib import Path
from typing import Union

import awkward as ak
import fastjet
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyjet
import pytorch_lightning as pl
import torch as T
import vector
from tqdm import tqdm

vector.register_awkward()


def locals_to_mass_and_pt(csts: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
    """Calculate the overall jet pt and mass from the constituents. The constituents are expected
    to be expressed as:

    - del_eta
    - del_phi
    - log_pt
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = csts[..., 2].exp()

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * T.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * T.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * T.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * T.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = T.clamp_min(jet_px**2 + jet_py**2, 0).sqrt()
    jet_m = T.clamp_min(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0).sqrt()

    return T.vstack([jet_pt, jet_m]).T


def numpy_locals_to_mass_and_pt(
    csts: np.ndarray,
    mask: np.ndarray,
    pt_logged=False,
) -> np.ndarray:
    """Calculate the overall jet pt and mass from the constituents. The constituents are expected
    to be expressed as:

    - del_eta
    - del_phi
    - log_pt or just pt depending on pt_logged
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = np.exp(csts[..., 2]) * mask if pt_logged else csts[..., 2]

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = np.sqrt(np.clip(jet_px**2 + jet_py**2, 0, None))
    jet_m = np.sqrt(np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None))

    return np.vstack([jet_pt, jet_m]).T


def dij(
    pt1: Union[float, np.float64, np.ndarray],
    pt2: Union[float, np.float64, np.ndarray],
    drs: Union[float, np.float64, np.ndarray],
    radius_par: float = 1.0,
    exp: int = 1,
) -> Union[float, np.float64, np.ndarray]:
    """Calculates pairs of dij values as implemented in jet cluster algos.

    Takes two unstructued ndarrays of transverse momenta, an ndarray of delta_r
    distance between object pairs and calculates the dij as used in the kt jet
    clustering algorithm. radius_par is the clustering radius parameter, exp the
    exponent (exp=1 kt, exp=0 C/A, exp=-1 anti-kt).

    Expects pt1, pt2 and drs to be of the same length.

    Args:
        pt1: First unstructured ndarray of pT values.
        pt2: Second unstructured ndarray of pT values.
        drs: Unstructured ndarray of delta_r values.
        radius_par: The clustering radius parameter. Default is radius_par=1.0.
        exp: The exponent used in the clustering. Default is exp=1.

    Returns:
        Unstructured array of dij values for objects pairs.
    """

    if type(pt1) is not type(pt2) or type(pt1) is not type(drs):
        raise TypeError("Inputs 'pt1', 'pt2' and 'drs' are not of the same type")
    if not isinstance(pt1, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if isinstance(pt1, np.ndarray) and isinstance(pt2, np.ndarray) and isinstance(drs, np.ndarray):
        if pt1.ndim != pt2.ndim or pt1.ndim != drs.ndim or pt1.ndim != 1:
            raise TypeError("Dimensions of input arrays are not equal to 1")
        if len(pt1) != len(pt2) or len(pt1) != len(drs):
            raise TypeError("Lengths of input arrays do not match")

    min_pt = np.amin((np.power(pt1, 2 * exp), np.power(pt2, 2 * exp)))
    return min_pt * drs * drs / radius_par / radius_par


def delta_r(
    eta1: Union[float, np.float64, np.ndarray],
    eta2: Union[float, np.float64, np.ndarray],
    phi1: Union[float, np.float64, np.ndarray],
    phi2: Union[float, np.float64, np.ndarray],
) -> Union[float, np.float64, np.ndarray]:
    """Calculates delta_r values between given ndarrays.

    Calculates the delta_r between objects. Takes unstructed ndarrays (or
    scalars) of eta1, eta2, phi1 and phi2 as input. Returns in the same format.
    This function can either handle eta1, phi1 to be numpy arrays (and eta2,
    phi2 to be floats), eta2, phi2 to be numpy arrays (and eta1, phi1 to be
    floats), or all four to be floats, and all four to be numpy arrays.
    Whenever numpy arrays are involved, they must be one-dimensional and of the
    same length.

    Args:
        eta1: First unstructured ndarray of eta values.
        eta2: Second unstructured ndarray of eta values.
        phi1: First unstructured ndarray of phi values.
        phi2: Second unstructured ndarray of phi values.

    Returns:
        Unstructured array of delta_r values between the pairs.
    """
    if type(eta1) is not type(phi1):
        raise TypeError("Inputs 'eta1' and 'phi1' must be of the same type")
    if type(eta2) is not type(phi2):
        raise TypeError("Inputs 'eta2' and 'phi2' must be of the same type")
    if not isinstance(eta1, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if not isinstance(eta2, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if isinstance(eta1, np.ndarray) and isinstance(phi1, np.ndarray):
        if eta1.ndim != 1 or phi1.ndim != 1:
            raise TypeError("Dimension of 'eta1' or 'phi1' is not equal to 1")
        if len(eta1) != len(phi1):
            raise TypeError("Lengths of 'eta1' and 'phi1' do not match")
    if isinstance(eta2, np.ndarray) and isinstance(phi2, np.ndarray):
        if eta2.ndim != 1 or phi2.ndim != 1:
            raise TypeError("Dimension of 'eta2' or 'phi2' is not equal to 1")
        if len(eta2) != len(phi2):
            raise TypeError("Lengths of 'eta2' and 'phi2' do not match")
    if isinstance(eta1, np.ndarray) and isinstance(eta2, np.ndarray) and len(eta1) != len(eta2):
        raise TypeError(
            "If 'eta1', 'eta2', 'phi1', 'phi2' are all of type np.ndarray, "
            "their lengths must be the same"
        )

    deta = np.absolute(eta1 - eta2)
    dphi = np.absolute(phi1 - phi2) % (2 * np.pi)
    dphi = np.min([2 * np.pi - dphi, dphi], axis=0)
    return np.sqrt(deta * deta + dphi * dphi)


def delta_r_min_to_axes(eta, phi, jet_axes):
    """Returns delta_r to closest jet axis for given eta/phi pair.

    Given unstructured ndarrays of eta and phi values and an unstructured
    ndarray of possible jet axes, finds the closest jet axis for each eta/phi
    pair in delta_r. Return the smallest delta_r value to any given axis, i.e.
    min(delta_r(cnst, axis1), delta_r(cnst, axis2), ...). Expects the jet_axes
    object to be of data type (_, eta, phi, _) per row.

    Args:
        eta: Unstructured ndarray of eta values.
        phi: Unstructured ndarray of phi values.
        jet_axes: Unstructured ndarray of jet axes, where eta and phi must be at
          index 1 and 2 per row, respectively (which corresponds to pyjet
          format).

    Returns:
        Unstructured ndarray with smallest delta_r obtained for each eta/phi pair.
    """
    if not isinstance(jet_axes, np.ndarray):
        raise TypeError("'jet_axes' needs to be of type np.ndarray")
    if jet_axes.ndim != 2:
        raise TypeError("np.ndarray 'jet_axes' needs to have dimension 2")
    if jet_axes.shape[1] != 4:
        raise TypeError("'jet_axes' needs to be of length '4' along the second axis")
    if type(eta) is not type(phi):
        raise TypeError("Inputs 'eta' and 'phi' must be of the same type")
    if not isinstance(eta, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if isinstance(eta, np.ndarray) and isinstance(phi, np.ndarray):
        if eta.ndim != 1 or phi.ndim != 1:
            raise TypeError("Dimension of 'eta' or 'phi' is not equal to 1")
        if len(eta) != len(phi):
            raise TypeError("Lengths of 'eta' and 'phi' do not match")
    delta_r_list = np.array([delta_r(eta, axis[1], phi, axis[2]) for axis in jet_axes])
    return np.amin(delta_r_list, axis=0)


class Substructure:
    """Calculates and holds substructure information per jet.

    This class calculates substructure observables for a pyjet.PseudoJet. Takes a PseudoJet object
    as input, calculates some essential re-clustering in the init function, then allows to retrieve
    various sorts of substructure variables through accessors. These are only calculated when
    called.
    """

    def __init__(self, jet, R):
        """Calculate essential reclustering for given pyjet.PseudoJet object.

        This retrieves the constituent particles of the jet, reshapes them into
        an array with axis0 = n_cnsts and axis1 = (pt, eta, phi, mass). Then
        reclusters constituent particles with the kt algorithm and stores lists
        of N-exclusive jets (exclusive kt clustering). Falls back to (N-1)
        clustering if there are not enough constituent particles for N.

        Args:
            jet: The pyjet.PseudoJet object.
            R: The jet radius used for reclustering.
        """
        R = 1.0 if R is None else R
        self._cnsts = jet.constituents_array()
        self._cnsts = self._cnsts.view(dtype=np.float64).reshape(self._cnsts.shape + (-1,))

        rclst = pyjet.cluster(jet.constituents_array(), R=R, p=1)
        self._subjets1 = np.array(
            [[_j.pt, _j.eta, _j.phi, _j.mass] for _j in rclst.exclusive_jets(1)]
        )
        try:
            self._subjets2 = np.array(
                [[_j.pt, _j.eta, _j.phi, _j.mass] for _j in rclst.exclusive_jets(2)]
            )
        except ValueError:
            self._subjets2 = self._subjets1
        try:
            self._subjets3 = np.array(
                [[_j.pt, _j.eta, _j.phi, _j.mass] for _j in rclst.exclusive_jets(3)]
            )
        except ValueError:
            self._subjets3 = self._subjets2

        # Store the frequently used sum of constituent transverse momenta.
        self._ptsum = np.sum(self._cnsts[:, 0])

    def d12(self):
        """Calculates the d12 splitting scale.

        Calculates the splitting scale for 2-jet exclusive clustering: one expects one of the jets
        in N-exclusive clustering to split in two in N+1-exclusive clustering. Locates these two
        'new' jets and returns the square root of their d_ij. If something goes wrong, default to
        0.
        """
        cmpl_indices = np.nonzero(
            np.isin(self._subjets2[:, 0], self._subjets1[:, 0], invert=True)
        )[0]
        if not len(cmpl_indices) == 2:
            return 0.0

        _j1 = self._subjets2[cmpl_indices[0]]
        _j2 = self._subjets2[cmpl_indices[1]]
        distance = dij(_j1[0], _j2[0], delta_r(_j1[1], _j2[1], _j1[2], _j2[2]))
        return 1.5 * np.sqrt(distance)

    def d23(self):
        """Calculates the d23 splitting scale.

        Calculates the splitting scale for 3-jet exclusive clustering: one expects one of the jets
        in N-exclusive clustering to split in two in N+1-exclusive clustering. Locates these two
        'new' jets and returns the square root of their d_ij. If something goes wrong, default to
        0.
        """
        cmpl_indices = np.nonzero(
            np.isin(self._subjets3[:, 0], self._subjets2[:, 0], invert=True)
        )[0]
        if not len(cmpl_indices) == 2:
            return 0.0

        _j1 = self._subjets3[cmpl_indices[0]]
        _j2 = self._subjets3[cmpl_indices[1]]
        distance = dij(_j1[0], _j2[0], delta_r(_j1[1], _j2[1], _j1[2], _j2[2]))
        return 1.5 * np.sqrt(distance)

    def ecf2(self):
        """Calculates the degree-2 energy correlation factor.

        Calculates the degree-2 energy correlation factor of the constituent
        particles. Takes transverse momenta and delta_r distances into account:

        >> sum (i<j in cnsts) pt(i) * pt(j) * delta_r(i, j)

        To avoid for-loop nesting, creates an array of unique index pairs, then
        uses those to access the constituent particles to be able to vectorise
        the operation. Internal function calc_ecf2(i, j) takes lists of
        components.
        """
        indices = np.arange(len(self._cnsts), dtype=np.uint8)
        idx_pairs = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2)
        idx_pairs = idx_pairs[(idx_pairs[:, 0] < idx_pairs[:, 1])]

        def calc_ecf2(i, j):
            return i[:, 0] * j[:, 0] * delta_r(i[:, 1], j[:, 1], i[:, 2], j[:, 2])

        return (
            calc_ecf2(self._cnsts[idx_pairs][:, 0], self._cnsts[idx_pairs][:, 1]).sum()
            / self._ptsum
            / self._ptsum
        )

    def ecf3(self):
        """Calculates the degree-3 energy correlation factor.

        Calculates the degree-3 energy correlation factor of the constituent
        particles. Takes transverse momenta and delta_r distances into account:

        >> sum (i<j<k in cnsts) pt(i) * pt(j) * pt(k)
        >>                      * delta_r(i, j) * delta_r(j, k) * delta_r(k, i)

        To avoid for-loop nesting, creates array of unique index triplets, then
        uses those to access the constituent particles to be able to vectorise
        the operation. Internal function calc_ecf2(i, j, k) takes lists of
        components.
        """
        indices = np.arange(len(self._cnsts), dtype=np.uint8)
        idx_pairs = np.array(np.meshgrid(indices, indices, indices)).T.reshape(-1, 3)
        idx_pairs = idx_pairs[
            (idx_pairs[:, 0] < idx_pairs[:, 1]) & (idx_pairs[:, 1] < idx_pairs[:, 2])
        ]

        def calc_ecf3(i, j, k):
            return (
                i[:, 0]
                * j[:, 0]
                * k[:, 0]
                * delta_r(i[:, 1], j[:, 1], i[:, 2], j[:, 2])
                * delta_r(j[:, 1], k[:, 1], j[:, 2], k[:, 2])
                * delta_r(k[:, 1], i[:, 1], k[:, 2], i[:, 2])
            )

        return (
            calc_ecf3(
                self._cnsts[idx_pairs][:, 0],
                self._cnsts[idx_pairs][:, 1],
                self._cnsts[idx_pairs][:, 2],
            ).sum()
            / self._ptsum
            / self._ptsum
            / self._ptsum
        )

    def tau1(self):
        """Calculates the 1-subjettiness.

        Calculates the 1-subjettiness (sum over minimal distances to jet axes for exclusive 1-jet
        clustering, weighted with constituent particle pT). Returns the dimensionless version of
        tau1, i.e., divided by the sum of all constituent transverse momenta.
        """
        dr_vals = delta_r_min_to_axes(self._cnsts[:, 1], self._cnsts[:, 2], self._subjets1)
        return np.sum(self._cnsts[:, 0] * dr_vals) / self._ptsum

    def tau2(self):
        """Calculates the 2-subjettiness.

        Calculates the 2-subjettiness (sum over minimal distances to jet axes for exclusive 2-jet
        clustering, weighted with constituent particle pT). Returns the dimensionless version of
        tau2, i.e., divided by the sum of all constituent transverse momenta.
        """
        dr_vals = delta_r_min_to_axes(self._cnsts[:, 1], self._cnsts[:, 2], self._subjets2)
        return np.sum(self._cnsts[:, 0] * dr_vals) / self._ptsum

    def tau3(self):
        """Calculates the 3-subjettiness.

        Calculates the 3-subjettiness (sum over minimal distances to jet axes for exclusive 3-jet
        clustering, weighted with constituent particle pT). Returns the dimensionless version of
        tau3, i.e., divided by the sum of all constituent transverse momenta.
        """
        dr_vals = delta_r_min_to_axes(self._cnsts[:, 1], self._cnsts[:, 2], self._subjets3)
        return np.sum(self._cnsts[:, 0] * dr_vals) / self._ptsum


def dump_hlvs(
    jets: np.ndarray,
    h5file: Path,
    R: float = 0.8,
    p: float = -1.0,
    plot: bool = False,
) -> None:
    """Given the nodes of a point cloud jet, compute the subtstructure variables and dump them to a
    file.

    Parameters
    ----------
    jets : np.ndarray
        jets represented as 3D particle clouds
        shape = (n_nodes, n_constituents, 3)
        The constituents are in the form (eta, phi, pt)
    h5file : Path
        Path to the h5 file to dump the substructure variables
    R : float, optional
        Jet radius, by default 0.8
    p : float, optional
        degree of the kt algorithm, by default -1.0
    plot : bool, optional
        Whether to plot the substructure variables, by default False
    Returns
    -------
    None
    """

    # Initialise the lists to hold the substructure variables
    tau_1s = []
    tau_2s = []
    tau_3s = []
    tau_21s = []
    tau_32s = []
    d12s = []
    d23s = []
    ecf2s = []
    ecf3s = []
    d2s = []
    d2s_new = []

    # Get the mask of the jets so that the padded elements dont contribute
    masks = np.any(jets != 0, axis=-1)

    # First load the pt and mass, variables often needed alongside substructure
    pt_mass = numpy_locals_to_mass_and_pt(jets, masks)
    pt = pt_mass[:, 0]
    mass = pt_mass[:, 1]

    # The substructure functions need data to be in [pt, eta, phi, m]
    jets = np.concatenate(
        [jets[..., [2, 0, 1]], np.zeros(shape=(*jets.shape[:-1], 1))],
        axis=-1,
    )
    jets = np.ascontiguousarray(jets)

    # Get the scalar sum of the pts (this will be the ecf1 variable)
    sum_pts = np.sum(jets[..., 0], axis=-1)

    # TODO Add particle net outputs pn_outs = partclenet()

    # Cycle through the jets and the sum pt values
    for jet, sum_pt, mask in tqdm(
        zip(jets, sum_pts, masks),
        total=len(jets),
        desc="Computing substructure variables",
    ):
        # pyjet needs each jet to be a be a structured array of type float64
        jet = jet[mask].view(
            [
                ("pt", np.float64),
                ("eta", np.float64),
                ("phi", np.float64),
                ("mass", np.float64),
            ]
        )
        incl_cluster = pyjet.cluster(jet, R=R, p=p)
        incl_jets = incl_cluster.inclusive_jets()[0]
        subs = Substructure(incl_jets, R=R)

        # NSubjettiness
        tau1 = subs.tau1()
        tau2 = subs.tau2()
        tau3 = subs.tau3()
        tau_1s.append(tau1)
        tau_2s.append(tau2)
        tau_3s.append(tau3)
        tau_21s.append(tau2 / tau1)
        tau_32s.append(tau3 / tau2)

        # Energy Splitting Functions
        d12s.append(subs.d12())
        d23s.append(subs.d23())

        # Energy Correlation Functions (first is simply the sum_pt of the jet)
        ecf2 = subs.ecf2()
        ecf3 = subs.ecf3()
        ecf2s.append(ecf2)
        ecf3s.append(ecf3)

        # ATLAS D2
        d2s.append((ecf3 * sum_pt) / (ecf2**2))
        d2s_new.append((ecf3) / (ecf2**3))

    # Save all the data to an HDF file
    with h5py.File(h5file + ".h5", mode="w") as file:
        file.create_dataset("tau1", data=tau_1s)
        file.create_dataset("tau2", data=tau_2s)
        file.create_dataset("tau3", data=tau_3s)
        file.create_dataset("tau21", data=tau_21s)
        file.create_dataset("tau32", data=tau_32s)
        file.create_dataset("d12", data=d12s)
        file.create_dataset("d23", data=d23s)
        file.create_dataset("ecf2", data=ecf2s)
        file.create_dataset("ecf3", data=ecf3s)
        file.create_dataset("d2", data=d2s)
        file.create_dataset("d2_new", data=d2s_new)
        file.create_dataset("pt", data=pt)
        file.create_dataset("mass", data=mass)

    # Include some plots
    if plot:
        fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8), (ax9, ax10, ax11)) = plt.subplots(
            4, 3, figsize=(15, 20)
        )
        ax0.hist(tau_1s, histtype="step", bins=100)
        ax0.set_title("tau1")
        ax1.hist(tau_2s, histtype="step", bins=100)
        ax1.set_title("tau2")
        ax2.hist(tau_3s, histtype="step", bins=100)
        ax2.set_title("tau3")
        ax3.hist(tau_21s, histtype="step", bins=100)
        ax3.set_title("tau21")
        ax4.hist(tau_32s, histtype="step", bins=100)
        ax4.set_title("tau32")
        ax5.hist(d12s, histtype="step", bins=100)
        ax5.set_title("d12")
        ax6.hist(d23s, histtype="step", bins=100)
        ax6.set_title("d23")
        ax7.hist(ecf2s, histtype="step", bins=100)
        ax7.set_title("ecf2")
        ax8.hist(ecf3s, histtype="step", bins=100)
        ax8.set_title("ecf3")
        ax9.hist(d2s, histtype="step", bins=100)
        ax9.set_title("d2")
        ax10.hist(pt, histtype="step", bins=100)
        ax10.set_title("pt")
        ax11.hist(mass, histtype="step", bins=100)
        ax11.set_title("mass")
        plt.tight_layout()
        fig.savefig(h5file + ".png")


# new implementation based on fastjet and awkward arrays

# substructure calculations
def calc_deltaR(particles, jet):
    jet = ak.unflatten(ak.flatten(jet), counts=1)
    return particles.deltaR(jet)


class JetSubstructure:
    """Class to calculate and store the jet substructure variables.

    Definitions as in slide 7 here:
    https://indico.cern.ch/event/760557/contributions/3262382/attachments/1796645/2929179/lltalk.pdf
    """

    def __init__(self, particles, R=1):
        """
        Parameters
        ----------
        particles : awkward array
            The particles that are clustered into jets.
        R : float
            The jet radius for the reclustering.
        """
        self.R = R
        self.particles = particles
        jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R)
        self.cluster = fastjet.ClusterSequence(particles, jetdef)
        self.inclusive_jets = self.cluster.inclusive_jets()
        self.exclusive_jets_1 = self.cluster.exclusive_jets(n_jets=1)
        self.exclusive_jets_2 = self.cluster.exclusive_jets(n_jets=2)
        self.exclusive_jets_3 = self.cluster.exclusive_jets(n_jets=3)

        self._calc_ptsum()
        self._calc_d0()
        print("Calculating N-subjettiness")
        self._calc_tau1()
        self._calc_tau2()
        self._calc_tau3()
        print("Calculating ECFs")
        # ECF1 is just the ptsum
        self._calc_ecf2()
        self._calc_ecf3()
        self.tau21 = self.tau2 / self.tau1
        self.tau32 = self.tau3 / self.tau2
        self.e2 = self.ecf2 / self.ptsum**2
        self.e3 = self.ecf3 / self.ptsum**3
        # D2 as defined in https://arxiv.org/pdf/1409.6298.pdf
        self.d2 = (self.e3) / self.e2**3

    def _calc_ptsum(self):
        """Calculate the ptsum values."""
        self.ptsum = ak.sum(self.particles.pt, axis=1)

    def _calc_d0(self):
        """Calculate the d0 values."""
        self.d0 = ak.sum(self.particles.pt * self.R, axis=1)

    def _calc_tau1(self):
        """Calculate the tau1 values."""
        self.delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_1[:, :1])
        self.pt_i = self.particles.pt
        # calculate the tau1 values
        self.tau1 = ak.sum(self.pt_i * self.delta_r_1i, axis=1) / self.d0

    def _calc_tau2(self):
        """Calculate the tau2 values."""
        delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_2[:, :1])
        delta_r_2i = calc_deltaR(self.particles, self.exclusive_jets_2[:, 1:2])
        self.pt_i = self.particles.pt
        # add new axis to make it broadcastable
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis],
                    delta_r_2i[..., np.newaxis],
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau2 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def _calc_tau3(self):
        """Calculate the tau3 values."""
        delta_r_1i = calc_deltaR(self.particles, self.exclusive_jets_3[:, :1])
        delta_r_2i = calc_deltaR(self.particles, self.exclusive_jets_3[:, 1:2])
        delta_r_3i = calc_deltaR(self.particles, self.exclusive_jets_3[:, 2:3])
        self.pt_i = self.particles.pt
        min_delta_r = ak.min(
            ak.concatenate(
                [
                    delta_r_1i[..., np.newaxis],
                    delta_r_2i[..., np.newaxis],
                    delta_r_3i[..., np.newaxis],
                ],
                axis=-1,
            ),
            axis=-1,
        )
        self.tau3 = ak.sum(self.pt_i * min_delta_r, axis=1) / self.d0

    def _calc_ecf2(self):
        """Calculate the ecf2 values."""
        particles_ij = ak.combinations(self.particles, 2, replacement=False)
        particles_i, particles_j = ak.unzip(particles_ij)
        delta_r_ij = particles_i.deltaR(particles_j)
        pt_ij = particles_i.pt * particles_j.pt

        self.ecf2 = ak.sum(pt_ij * delta_r_ij, axis=1)

    def _calc_ecf3(self):
        """Calculate the ecf3 values."""
        particles_ijk = ak.combinations(self.particles, 3, replacement=False)
        particles_i, particles_j, particles_k = ak.unzip(particles_ijk)
        delta_r_ij = particles_i.deltaR(particles_j)
        delta_r_ik = particles_i.deltaR(particles_k)
        delta_r_jk = particles_j.deltaR(particles_k)
        pt_ijk = particles_i.pt * particles_j.pt * particles_k.pt

        self.ecf3 = ak.sum(pt_ijk * delta_r_ij * delta_r_ik * delta_r_jk, axis=1)


def calc_substructure(
    particles_sim,
    particles_gen,
    R=1,
    filename=None,
):
    """Calculate the substructure variables for the given particles and save them to a file.

    Parameters
    ----------
    particles_sim : awkward array
        The particles of the simulated jets.
    particles_gen : awkward array
        The particles of the generated jets.
    R : float, optional
        The jet radius, by default 1
    filename : str, optional
        The filename to save the results to, by default None (don't save)
    """
    if filename is None:
        print("No filename given, won't save the results.")
    else:
        if os.path.exists(filename):
            print(f"File {filename} already exists, won't overwrite.")
            return
        print(f"Saving results to {filename}")

    substructure_sim = JetSubstructure(particles_sim, R=R)
    substructure_gen = JetSubstructure(particles_gen, R=R)
    names = [
        "tau1",
        "tau2",
        "tau3",
        "tau21",
        "tau32",
        "ecf2",
        "ecf3",
        "e2",
        "e3",
        "d2",
    ]
    with h5py.File(filename, "w") as f:
        for name in names:
            f[f"{name}_sim"] = substructure_sim.__dict__[name]
            f[f"{name}_gen"] = substructure_gen.__dict__[name]
