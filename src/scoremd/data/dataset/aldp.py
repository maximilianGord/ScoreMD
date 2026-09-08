from dataclasses import dataclass
from os import PathLike
from typing import Optional, Tuple, Callable
from scoremd.data.dataset import Dataset
import jax
import jax.numpy as jnp
import numpy as onp
import logging
import openmm as mm
from scoremd.data.dataset.angles import dihedral
from scoremd.data.dataset.base import Datapoints
from scoremd.data.dataset.utils import write_animation_with_topology
from scoremd.rmsd import kabsch_align_many
from scoremd.simulation import create_langevin_step_function
from scoremd.utils.file import get_persistent_storage
from bgmol.datasets import AImplicitUnconstrained
import openmm.unit as unit
from scoremd.utils.openmm import get_fastest_platform
from enum import Enum
import scoremd.utils.plots as plots

log = logging.getLogger(__name__)


class CoarseGrainingLevel(Enum):
    NONE = "NONE"
    REMOVE_HYDROGENS = "REMOVE_HYDROGENS"
    SIX_BEADS = "SIX_BEADS"
    # We keep 5 beads
    FULL = "FULL"


@dataclass
class ALDPDataset(Dataset):
    """This is a dataset of alanine dipeptide in implicit solvent. All coordinates are in nanometers."""

    coarse_graining_level: CoarseGrainingLevel = CoarseGrainingLevel.NONE
    train_split: Tuple[float, float] = 0.8
    limit_samples: Optional[int] = None
    validation: bool = False
    test: bool = False
    path: Optional[PathLike] = None  # Can be used to load a custom dataset
    seed: int = 0

    def __init__(
        self,
        coarse_graining_level: CoarseGrainingLevel | str = CoarseGrainingLevel.NONE,
        path: Optional[PathLike] = None,
        train_split: float = 0.8,
        limit_samples: Optional[int] = None,
        validation: bool = True,
        seed: int = 0,
        name="aldp",
    ):
        self.train_split = train_split
        self.limit_samples = limit_samples
        self.validation = validation
        self.seed = seed
        self._dataset = None
        self._path = path
        self._train_force_coordinates = None
        self._val_force_coordinates = None

        root = get_persistent_storage()
        try:
            log.debug("Trying to load dataset from cache.")
            self._dataset = AImplicitUnconstrained(root=root, download=False, read=True)
        except FileNotFoundError as e:
            log.warning(f"Failed to load dataset: {e}")
            log.info("Downloading dataset...")
            self._dataset = AImplicitUnconstrained(root=root, download=True, read=True)

        platform = get_fastest_platform()
        if platform.getName() == "CUDA":
            platform.setPropertyDefaultValue("Precision", "mixed")

        self.context = mm.Context(self._dataset.system.system, self._dataset.integrator, platform)

        self.coarse_graining_level = (
            coarse_graining_level
            if isinstance(coarse_graining_level, CoarseGrainingLevel)
            else CoarseGrainingLevel(coarse_graining_level.upper())
        )
        if self.coarse_graining_level == CoarseGrainingLevel.NONE:
            self._atoms_to_keep = list(range(22))
        elif self.coarse_graining_level == CoarseGrainingLevel.REMOVE_HYDROGENS:
            self._atoms_to_keep = [0, 4, 5, 6, 8, 10, 14, 15, 16, 18]
        elif self.coarse_graining_level == CoarseGrainingLevel.SIX_BEADS:
            self._atoms_to_keep = [4, 6, 8, 10, 14, 16]
        elif self.coarse_graining_level == CoarseGrainingLevel.FULL:
            self._atoms_to_keep = [4, 6, 8, 14, 16]
        else:
            raise ValueError("Unknown coarse graining level.")

        log.debug(f"Keeping atoms: {self._atoms_to_keep}")

        self.mass = jnp.array(
            [
                self._dataset.system.system.getParticleMass(i).value_in_unit(unit.dalton)
                for i in range(self._dataset.system.system.getNumParticles())
            ]
        ).reshape(-1, 1)[self._atoms_to_keep, ...]

        temp = self._dataset.integrator.getTemperature()

        super().__init__(
            name=name,
            sample_shape=(len(self._atoms_to_keep), 3),
            kbT=(unit.MOLAR_GAS_CONSTANT_R * temp).value_in_unit(unit.kilojoules_per_mole),
        )

    def _split_indices(self, num_samples: int, key: jnp.ndarray) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        indices = jax.random.permutation(key, num_samples)
        if self.validation:
            train_set = round(num_samples * self.train_split)
            assert train_set > 0, "No training samples."
            train_indices, val_indices = indices[:train_set], indices[train_set:]
        else:
            train_indices, val_indices = indices, indices[:0]

        if self.limit_samples:
            new_train_set_size = min(train_indices.shape[0], self.limit_samples)
            new_val_set_size = int(jnp.ceil(new_train_set_size / self.train_split * (1 - self.train_split)))
            log.info(
                f"Limit samples has been set to: {self.limit_samples}. "
                f"Limiting training dataset to {new_train_set_size} and validation set to {new_val_set_size} sample(s)."
            )
            train_indices = train_indices[:new_train_set_size]
            val_indices = val_indices[:new_val_set_size]

        return train_indices, val_indices if val_indices.shape[0] > 0 else None

    @staticmethod
    def _align_full_coordinates(
        full_coordinates: onp.ndarray, coarse_coordinates: jnp.ndarray, rotations: jnp.ndarray
    ) -> onp.ndarray:
        """Apply the CG Kabsch centering and rotation to matching full-atom frames."""
        coarse_coordinates = onp.asarray(coarse_coordinates).reshape((-1, coarse_coordinates.shape[-1] // 3, 3))
        full_coordinates = onp.asarray(full_coordinates).reshape((-1, 22, 3))
        centered_full_coordinates = full_coordinates - onp.mean(coarse_coordinates, axis=1, keepdims=True)
        aligned = onp.matmul(centered_full_coordinates, onp.swapaxes(onp.asarray(rotations), -1, -2))
        return aligned.reshape((aligned.shape[0], -1))

    def force_coordinates_for(self, datapoints: Datapoints) -> Optional[onp.ndarray]:
        """Return full-atom coordinates for one-time projected-force evaluation."""
        force_coordinates = None
        if datapoints is self._train:
            force_coordinates = self._train_force_coordinates
        elif datapoints is self._val:
            force_coordinates = self._val_force_coordinates
        if force_coordinates is None and self.coarse_graining_level != CoarseGrainingLevel.NONE:
            raise ValueError(
                "Projected TSM/SC forces require paired full-atom ALDP coordinates; "
                "a custom coarse-grained path cannot provide them."
            )
        return force_coordinates

    def release_force_coordinates(self, datapoints: Datapoints) -> None:
        """Discard full-atom coordinates once their projected force targets are materialized."""
        if datapoints is self._train:
            self._train_force_coordinates = None
        elif datapoints is self._val:
            self._val_force_coordinates = None

    def project_forces(self, full_forces: jnp.ndarray) -> jnp.ndarray:
        """Project 22-atom OpenMM forces onto the atoms represented by this dataset."""
        full_forces = jnp.asarray(full_forces).reshape((-1, 3))
        if full_forces.shape[0] != 22:
            raise ValueError(f"Expected forces for 22 atoms, got {full_forces.shape[0]}.")
        return full_forces[jnp.asarray(self._atoms_to_keep, dtype=jnp.int32)].reshape(-1)

    def _get_data(self) -> Tuple[Datapoints, Optional[Datapoints], Optional[Datapoints]]:
        full_coordinates = None
        if self._path is not None:
            coarse_coordinates = jnp.asarray(onp.load(self._path))
            assert coarse_coordinates.shape[1] == len(self._atoms_to_keep), "Invalid number of atoms"
        else:
            full_coordinates = onp.asarray(self._dataset.xyz)
            coarse_coordinates = jnp.asarray(full_coordinates[:, self._atoms_to_keep, ...])

        train_indices, val_indices = self._split_indices(coarse_coordinates.shape[0], jax.random.PRNGKey(self.seed))
        train_unaligned = coarse_coordinates[train_indices].reshape((train_indices.shape[0], -1))
        train, _, train_rotations = kabsch_align_many(train_unaligned, train_unaligned[0], return_rotation=True)
        if full_coordinates is not None:
            self._train_force_coordinates = self._align_full_coordinates(
                full_coordinates[onp.asarray(train_indices)], train_unaligned, train_rotations
            )

        val = None
        if val_indices is not None:
            val_unaligned = coarse_coordinates[val_indices].reshape((val_indices.shape[0], -1))
            val, _, val_rotations = kabsch_align_many(val_unaligned, train[0], return_rotation=True)
            if full_coordinates is not None:
                self._val_force_coordinates = self._align_full_coordinates(
                    full_coordinates[onp.asarray(val_indices)], val_unaligned, val_rotations
                )

        return Datapoints(train, None), Datapoints(val, None) if val is not None else None, None

    def force(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Use openmm to calculate the forces acting on the system.
        The unit of x is nanometers and returns forces in the kJ/(nm mol) unit.
        """
        x = x.reshape(-1, 3)
        assert x.shape[0] == 22, f"Expected 22 atoms, got {x.shape[0]}."
        self.context.setPositions(x * unit.nanometers)
        state = self.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        return jnp.array(forces.in_units_of(unit.kilojoules_per_mole / unit.nanometers))

    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Use openmm to calculate the potential energy of the system.
        The unit of x is nanometers and returns energy in the kJ/mol unit.
        """
        x = x.reshape(-1, 3)
        assert x.shape[0] == 22, f"Expected 22 atoms, got {x.shape[0]}."
        self.context.setPositions(x * unit.nanometers)
        state = self.context.getState(getEnergy=True)
        return jnp.array(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

    def langevin_step_function(
        self,
        num_steps,
        force: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        dt: Optional[float] = None,
    ) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        This function returns a Langevin step function that takes nm coordinates and does num_steps.
        See the `create_langevin_step_function` function for more details.
        """
        return create_langevin_step_function(
            force=self.force if force is None else force,
            mass=self.mass,
            gamma=self._dataset.integrator.getFriction().value_in_unit(unit.picosecond**-1),
            num_steps=num_steps,
            dt=dt if dt is not None else self._dataset.integrator.getStepSize().value_in_unit(unit.picoseconds),
            kbT=self.kbT,
        )

    def get_2d_features(self, trajectory: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.coarse_graining_level == CoarseGrainingLevel.NONE:
            phi_indices, psi_indices = [4, 6, 8, 14], [6, 8, 14, 16]
        elif self.coarse_graining_level == CoarseGrainingLevel.REMOVE_HYDROGENS:
            phi_indices, psi_indices = [1, 3, 4, 6], [3, 4, 6, 8]
        elif self.coarse_graining_level == CoarseGrainingLevel.SIX_BEADS:
            phi_indices, psi_indices = [0, 1, 2, 4], [1, 2, 4, 5]
        elif self.coarse_graining_level == CoarseGrainingLevel.FULL:
            phi_indices, psi_indices = [0, 1, 2, 3], [1, 2, 3, 4]
        else:
            raise ValueError("Unknown coarse graining level.")

        # we jit it to save on GPU memory when reshaping
        @jax.jit
        def _get_2d_features(trajectory):
            trajectory = trajectory.reshape(trajectory.shape[0], -1, 3)  # data in nm
            return dihedral(trajectory[:, phi_indices]), dihedral(trajectory[:, psi_indices])

        return _get_2d_features(trajectory)

    def plot_2d(
        self, data, title="Histogram", highlight=None, bins=60, vmin=None, vmax=None, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        phi, psi = self.get_2d_features(data)

        plots.phi_psi(phi, psi, title, highlight, bins, vmin=vmin, vmax=vmax, **kwargs)
        return phi, psi

    def write_animation(self, trajectory: jnp.ndarray, out: PathLike):
        write_animation_with_topology(trajectory, self.topology, out)

    @property
    def bonds(self) -> set[Tuple[int, int]]:
        return {
            (b.atom1.index, b.atom2.index) if b.atom1.index < b.atom2.index else (b.atom2.index, b.atom1.index)
            for b in self.topology.bonds()
        }

    @property
    def atom_names(self):
        return [a.name for a in self.topology.atoms()]

    @property
    def topology(self) -> mm.app.Topology:
        topology = self._dataset.system.topology

        if self.coarse_graining_level == CoarseGrainingLevel.NONE:
            return topology

        # create a new topology with only the atoms we want to keep
        atoms = list(topology.atoms())
        filtered_atoms = {atoms[i] for i in self._atoms_to_keep}
        atom_mapping = {}  # Maps original atoms to new atoms in filtered_topology

        filtered_topology = mm.app.Topology()
        for chain in topology.chains():
            new_chain = None  # we will use this to only add the chain if any atom of any residue is kept
            for residue in chain.residues():
                kept_atoms = [atom for atom in residue.atoms() if atom in filtered_atoms]
                if len(kept_atoms) == 0:
                    continue

                if new_chain is None:  # first atom of the residue that is kept
                    new_chain = filtered_topology.addChain(chain.id)

                new_residue = filtered_topology.addResidue(residue.name, new_chain, residue.id)
                for atom in kept_atoms:
                    new_atom = filtered_topology.addAtom(atom.name, atom.element, new_residue)
                    atom_mapping[atom] = new_atom

        for bond in topology.bonds():
            a1, a2 = bond
            if a1 in filtered_atoms and a2 in filtered_atoms:
                filtered_topology.addBond(atom_mapping[a1], atom_mapping[a2])

        return filtered_topology
