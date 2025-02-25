from abc import ABC, abstractmethod
from typing import Sequence

Numpy1DArray = Sequence
Numpy2DArray = Sequence


class AtomicCodeIO(ABC):

    @abstractmethod
    def read_full_basis(self) -> Numpy2DArray:
        pass


    @abstractmethod
    def read_start_basis(self) -> Numpy2DArray:
        pass


    @abstractmethod
    def read_start_weights(self) -> Numpy1DArray[float]:
        pass


    @abstractmethod
    def read_current_weights(self) -> Numpy1DArray[float]:
        pass


    @abstractmethod
    def write_current_basis(self, which_write: Numpy1DArray[bool]) -> None:
        pass
