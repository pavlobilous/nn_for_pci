from abc import ABC, abstractmethod
from typing import Sequence

Numpy1DArray = Sequence
Numpy2DArray = Sequence


class AtomicCodeIO(ABC):
    """An interface for creating custom couplings to atomic codes."""

    @abstractmethod
    def read_full_basis(self) -> Numpy2DArray:
        """Takes no input and returns a 2D NumPy array for the full CI basis set with rows corresponding
        to relativistic configurations and columns representing relativistic orbital populations."""
        pass


    @abstractmethod
    def read_prior_basis(self) -> Numpy2DArray:
        """Takes no input and returns a 2D NumPy array for the "prior" CI basis set with rows corresponding
        to relativistic configurations and columns representing relativistic orbital populations."""
        pass


    @abstractmethod
    def read_prior_weights(self) -> Numpy1DArray[float]:
        """Takes no input and returns a 1D NumPy array with weights obtained in the "prior" computation."""
        pass


    @abstractmethod
    def read_current_weights(self) -> Numpy1DArray[float]:
        """Takes no input and returns a 1D NumPy array with weights obtained in the "current" CI run."""
        pass


    @abstractmethod
    def write_current_basis(self, which_write: Numpy1DArray[bool]) -> None:
        """Writes input for the next CI run.
        The argument which_write is a 1D boolean NumPy array with True
        corresponding to the relativistic configurations to be written."""
        pass
