The Python package `nn_for_pci` implements tools for configuration interaction (CI) computations with neural-network (NN) support.

The package consists of two subpackages: `neural_manager` and `pci_io`. The former encapsulates tools for the NN-based CI algorithm described in Ref. [1], whereas the latter implements the coupling to the pCI atomic codes [2] for solving the CI problem.

The main classes provided for an NN-supported CI computation are `NeuralManager` (from the `neural_manager` subpackage) and `PciIO` (from the `pci_io` subpackage). All necessary components can be imported from the top-level package `nn_for_pci`.

The user creates an instance of the `PciIO` class to establish the communication between the NN-part and the pCI atomic codes. This can be done using an auxiliary named tuple class `PciIOFiles` with the fields `conf_inp_full`, `conf_inp_prior`, `conf_res_prior`, `conf_inp_current`, `conf_res_current`. The user provides correspondingly the paths of the following files:
- CONF.INP file containing the full (large) set of relativistic configurations;
- CONF.INP file containing the set of relativistic configurations in the “prior” computation;
- CONF.RES file containing the CI expansion weights resulting from the “prior” computation;
- CONF.INP file where the NN-part writes the "current" input to the pCI code;
- CONF.RES file where the NN-part reads the "current" input to the pCI code.

Here, the “prior” computation is a smaller-scale CI computation performed directly (without NN). The created named tuple is used directly in the `PciIO` constructor. The obtained `PciIO` object, in turn, is used to create an instance of the `NeuralManager` class.

The `NeuralManager` class contains the following methods and properties for performing the iterative algorithm from Ref. [1].

`NeuralManager(code_io)` -- the constructor of the class.\
`code_io` is an instance of the `PciIO` class (or more generally the `AtomicCodeIO` class, see below).

`NeuralManager.start_new_comp(rand_frac, cutlog)` -- is used for starting a NN-supported CI computation by forming an input for the first pCI run (the results of the latter can be then used to train the NN)
- `rand_frac` is the fraction of relativistic configurations to be randomly included on top of the "prior" set;
- `cutlog` (key-only optional argument) is log_10 of the cutoff for relativistic configurations from the "prior" set to be included. If not specified, the whole "prior" set is included.

`NeuralManager.neural_sortout(cutlog, bal_ratio, nn_model, start_eval_kwargs, train_kwargs, apply_kwargs)` -- the method used for iterative NN-based sortout of the basis (this is the central function of the package)
- `cutlog` is log_10 of the "importance" cutoff for relativistic configurations. The NN will target inclusion of configuration with weights larger than this cutoff;
- `bal_ratio` is the fraction of NN-discarded configurations to be included with respect to the size of the set suggested by the NN as important;
- `nn_model` is a compiled Keras model;
- `start_eval_kwargs` is a dictionary with parameters controlling the initial NN evaluation (passed to the model's `evaluate` method);
- `train_kwargs` is a dictionary with parameters used for the NN training (passed to the model's `fit` method);
- `apply_kwargs` is a dictionary with parameters used for the NN classification of unknown relativistic configurations (passed to the model's `predict` method).

`NeuralManager.save_comp(path)` -- saves the NumPy arrays tracking the configuration selection process in order to perform a CI computation with pCI.

`NeuralManager.load_comp(path)` -- loads the NumPy arrays tracking the configuration selection process after a CI computation with pCI is finished.

`NeuralManager.full_basis_size` -- returns the full CI basis size.

`NeuralManager.features_num` -- returns the number of features, that is parameters used as NN input for each relativistic configurations.

It is possible to use the NN support with a different CI solver by replacing `PciIO` with a custom class implementing the corresponding coupling. This custom class must inherit from the abstract class `AtomicCodeIO` with implementation of its five abstract methods for writing the input for the atomic code and reading its output: `read_full_basis`, `read_start_basis`, `read_start_weights`, `read_current_weights`, `write_current_basis`.

Please follow the detailed instructions in Ref. [3] for performing NN-supported CI computations with the algorithm from Ref. [1].

[1] P. Bilous, C. Cheung, M. Safronova, "Neural-network approach to running high-precision atomic computations", Phys. Rev. A 110, 042818 (2024).\
[2] C. Cheung, M. G. Kozlov, S. G. Porsev, M. S. Safronova, I. I. Tupitsyn, A. I. Bondarev, "pCI: A parallel configuration interaction software package for high-precision atomic structure calculations", Computer Physics Communications 308, 109463 (2025).\
[3] P. Bilous, C. Cheung, M. Safronova, "A neural-network-based Python package for performing large-scale atomic CI using pCI and other high-performance atomic codes", arXiv:2503.01379 (2025).
