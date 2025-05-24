from abc import ABC, abstractmethod
from typing import Sequence

Numpy1DArray = Sequence
Numpy2DArray = Sequence


class AtomicCodeIO(ABC):
    """An interface for creating custom couplings to atomic codes."""

    @abstractmethod
    def read_full_basis(self) -> Numpy2DArray:
        pass


    @abstractmethod
    def read_prior_basis(self) -> Numpy2DArray:
        pass


    @abstractmethod
    def read_prior_weights(self) -> Numpy1DArray[float]:
        pass


    @abstractmethod
    def read_current_weights(self) -> Numpy1DArray[float]:
        pass


    @abstractmethod
    def write_current_basis(self, which_write: Numpy1DArray[bool]) -> None:
        pass
