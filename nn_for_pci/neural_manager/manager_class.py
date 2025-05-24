import numpy as np

from .atomic_code_io import AtomicCodeIO


class NeuralManager:
    """The NeuralManager class implements tools for NN-based CI computations.
    An instance is created using an instance of the PciIO class,
    or more generally of a class implementing the interface AtomicCodeIO."""

    @property
    def full_basis_size(self):
        """Number of relativistic configurations in the dataset."""
        return len(self._full_basis)


    @property
    def features_num(self):
        """Number of features in the dataset."""
        return self._full_basis.shape[1]


    def __init__(self, code_io: AtomicCodeIO):
        if not isinstance(code_io, AtomicCodeIO):
            raise TypeError("Atomic code interface must subclass AtomicCodeIO.")

        print("\n************************")
        print("Creating a NeuralManager")
        self._code_io = code_io

        print("==> Loading full basis...")
        self._full_basis = self._code_io.read_full_basis()
        print(f"\tFull basis size: {self.full_basis_size}")
        print(f"\tFeatures: {self.features_num}")
        print("Done.")


    def start_new_comp(self,
                       rand_frac: float,
                       *,
                       cutlog: float = None):
        """Is used for starting a NN-supported CI computation by forming an input for the first atomic code run
        - rand_frac is the fraction of relativistic configurations
            to be randomly included on top of the "prior" set;
        - cutlog (key-only optional argument) is log_10 of the cutoff for relativistic configurations
            from the "prior" set to be included. If not specified, the whole "prior" set is included."""
        from .start_new_comp import \
                    create_state_arrs, start_fill, add_randoms
        print("\nStarting a new neural-network-supported computation")
        self._state_arrs = create_state_arrs(self.full_basis_size)

        print("==> Loading prior basis...")
        prior_basis = self._code_io.read_prior_basis()
        prior_weights = self._code_io.read_prior_weights()
        print(f"\tPrior basis size: {len(prior_basis)}")
        with_cutlog = (cutlog is not None)
        if with_cutlog:
            print(f"\tcutlog: {cutlog}")
        else:
            cutlog = -np.inf
        impt_num = start_fill(cutlog,
                    prior_basis, prior_weights,
                    self._full_basis, self._state_arrs)
        if with_cutlog:
            print(f"\tRemained after cut: {impt_num}")
        print("Done.")

        print("==> Adding randoms on top...")
        print(f"\tFraction of randoms: {rand_frac}")
        rand_num = add_randoms(rand_frac, self._state_arrs)
        print(f"\tNumber of randoms: {rand_num}")
        print("Done.")
        
        print(f"==> Writing input for the atomic code...")
        print(f"\tNumber of written: {self._state_arrs['onoff'].sum()}")
        self._code_io.write_current_basis(self._state_arrs["onoff"])
        print("Done.")

        return prior_basis



    def neural_sortout(self,
                       cutlog: float,
                       bal_ratio: float,
                       nn_model,
                       start_eval_kwargs: dict,
                       train_kwargs: dict,
                       apply_kwargs: dict):
        """Used for iterative NN-based sortout of the basis.
        - cutlog is log_10 of the "importance" cutoff for relativistic configurations;
        - bal_ratio is the fraction of NN-discarded configurations to be included
            with respect to the size of the set suggested by the NN as important;
        - nn_model is a compiled Keras model;
        - start_eval_kwargs is a dictionary with parameters controlling the initial NN evaluation;
        - train_kwargs is a dictionary with parameters used for the NN training;
        - apply_kwargs is a dictionary with parameters used for the NN classification
            of unknown relativistic configurations."""
        from .neural_sortout import train_nn, apply_nn, balance, toggle_state_arrs

        state_arrs = self._state_arrs
        full_basis = self._full_basis

        print("\nSorting out the basis using a neural network")
        print(f"cutlog: {cutlog}")

        print("==> Reading and preparing weights...")
        weights = state_arrs.pop("prior_weights", np.zeros(len(full_basis)))
        if weights.any():
            print("\tPrior weights were present;")
            print("\tthey are taken into account and deleted.")
        
        weights[state_arrs["onoff"]] = self._code_io.read_current_weights()
        print("Done.")

        print("==> Training the neural network...")
        print(f"\tTraining set size: {state_arrs['train'].sum()}")
        print(f"\t...of which important: {(weights[state_arrs['train']] > 10**cutlog).sum()}")
        train_nn(state_arrs, full_basis, weights, cutlog, nn_model, start_eval_kwargs, train_kwargs)
        print("Done.")

        print("==> Applying the neural network...")
        print(f"\tApplication to: {state_arrs['apply'].sum()}")
        predimp_inds, prednotimp_inds = apply_nn(state_arrs, full_basis, nn_model, apply_kwargs)
        print(f"\tClassified as important: {len(predimp_inds)}")
        print("Done.")

        print("==> Balancing the next training set...")
        print(f"\tBalancing fraction: {bal_ratio}")
        bal_inds = balance(predimp_inds, prednotimp_inds, bal_ratio)
        print(f"\tBalance set size: {len(bal_inds)}")
        print("Done.")

        print(f"==> Writing input for the atomic code...")
        toggle_state_arrs(state_arrs, weights, cutlog, predimp_inds, bal_inds)
        print(f"\tNumber of written: {state_arrs['onoff'].sum()}")
        self._code_io.write_current_basis(self._state_arrs["onoff"])
        print("Done.")


    def load_comp(self, path):
        """Loads a NN-supported CI computation saved at "path".
        Note that this does not load TensorFlow NN models."""
        from .save_load import load_state_arrs
        print("\nLoading computation")
        print(f"\tpath='{path}'")
        self._state_arrs = load_state_arrs(path)
        print("Done.")



    def save_comp(self, path):
        """Saves a NN-supported CI computation saved to "path".
        Note that this does not save TensorFlow NN models."""
        from .save_load import save_state_arrs
        print("\nSaving computation")
        print(f"\tpath='{path}'")
        save_state_arrs(path, self._state_arrs)
        print("Done.")
