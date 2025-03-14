The Python package `nn_for_pci` offers tools for configuration interaction (CI) computations with neural-network (NN) support.

The package consists of two subpackages: `neural_manager` and `pci_io`. The former encapsulates tools for the NN-based CI algorithm described in Ref. [1], whereas the latter implements the coupling to the pCI atomic codes [2] for solving the CI problem.

The main classes provided for an NN-supported CI computation are `NeuralManager` (from the `neural_manager` subpackage) and `PciIO` (from the `pci_io` subpackage). Please follow the instructions in Ref. [3] for performing NN-supported CI computations with the algorithm from Ref. [1].

It is possible to use the NN support with a different CI solver by replacing `PciIO` with a custom class implementing the corresponding coupling. This custom class must inherit from the abstract class `AtomicCodeIO` with implementation of its five abstract methods for writing the input for the atomic code and reading its output. Please follow the instructions in Ref. [3].

[1] P. Bilous, C. Cheung, M. Safronova, "Neural-network approach to running high-precision atomic computations", Phys. Rev. A 110, 042818 (2024).
[2] C. Cheung, M. G. Kozlov, S. G. Porsev, M. S. Safronova, I. I. Tupitsyn, A. I. Bondarev, "pCI: A parallel configuration interaction software package for high-precision atomic structure calculations", Computer Physics Communications 308, 109463 (2025).
[3] P. Bilous, C. Cheung, M. Safronova, "A neural-network-based Python package for performing large-scale atomic CI using pCI and other high-performance atomic codes", arXiv:2503.01379 (2025).
