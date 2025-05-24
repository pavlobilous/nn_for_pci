from collections import namedtuple
from typing import Callable, Sequence
from types import MappingProxyType
import numpy as np

from ..neural_manager import AtomicCodeIO
from .read_files import *


weights_aggregator = lambda arr: arr.max(axis=1)

PciIOFiles = namedtuple("PciIOFiles",
            "conf_inp_full conf_inp_prior conf_res_prior conf_inp_current conf_res_current".split())

def _digitize_arr(arr):
    unp =  np.unpackbits(arr, axis=1)
    useless = (unp == 1).all(axis=0) | (unp == 0).all(axis=0)
    return unp[:, ~useless]


def _normalize_arr(arr):
    norm = arr.max(axis=0)
    norm[np.where(norm == 0)[0]] = 1
    res = np.divide(arr, norm, dtype=np.float32)
    return res


class PciIO(AtomicCodeIO):
    """PciIO class is used to establish the communication between the NN-part and the pCI atomic codes.
    
    PciIO instance is created using an auxiliary named tuple class PciIOFiles with the fields:
    - conf_inp_full,
    - conf_inp_prior,
    - conf_res_prior,
    - conf_inp_current,
    - conf_res_current,
    where the paths of the following files are provided:
    - CONF.INP file containing the full (large) set of relativistic configurations;
    - CONF.INP file containing the set of relativistic configurations in the “prior” computation;
    - CONF.RES file containing the CI expansion weights resulting from the “prior” computation;
    - CONF.INP file where the NN-part writes the "current" input to the pCI code;
    - CONF.RES file where the NN-part reads the "current" input to the pCI code.
    
    Additionally, arguments `digitize` and `normalize` can be provided (both by default False, cannot be True at the same time),
    which trigger a corresponding transformation of the dataset with relativistic configurations immediately after the loading."""

    def __init__(self, io_files: PciIOFiles,
                 *,
                 digitize: bool = False,
                 normalize: bool = False):
        self.io_files = io_files

        with open(self.io_files.conf_inp_full, "r") as f:
            header_gen = read_conf_inp_header_lines(f)
            params = {}
            for _, p in header_gen:
                params.update(p)
            self.num_levels = params["Nlv"]
            self.num_confs_full = params["Nc"]
            self.num_stat_orbs = params["Nso"]

            self.orbs = {}
            confs_gen = read_conf_inp_confs_lines(
                    f, self.num_confs_full, self.num_stat_orbs,
                    self.orbs
                )
            for _ in confs_gen:
                pass

        if digitize and normalize:
            raise ValueError('"digitize" and "normalize" agruments are mutually exclusive and cannot be True at the same time.')

        if digitize:
            self.transform = "digitize"
        elif normalize:
            self.transform = "normalize"
        else:
            self.transform = None

        self.weights_aggregator = weights_aggregator

    
    #override
    def read_full_basis(self):
        conf_dest_arr = np.zeros((self.num_confs_full, len(self.orbs)), dtype=np.uint8)
        nonrel_dest_arr = np.zeros(self.num_confs_full, dtype=int)
        with open(self.io_files.conf_inp_full, "r") as f:
            header_gen = read_conf_inp_header_lines(f)
            for _ in header_gen:
                pass
            confs_gen = read_conf_inp_confs_lines(
                    f, self.num_confs_full, self.num_stat_orbs,
                    MappingProxyType(self.orbs), conf_dest_arr, nonrel_dest_arr
                )
            for _ in confs_gen:
                pass
        self.nonrel_groups = nonrel_dest_arr
        if self.transform == "digitize":
            conf_dest_arr = _digitize_arr(conf_dest_arr)
        elif self.transform == "normalize":
            conf_dest_arr = _normalize_arr(conf_dest_arr)
        else:
            if self.transform is not None:
                raise ValueError('The "transform" attribute must be "digitize", "normalize" or None (default).')
        return conf_dest_arr


    #override
    def read_prior_basis(self):
        if self.transform is not None:
            raise RuntimeError("The prior basis can't be processed if the full basis features are transformed (digitized or normalized). "\
                               "Create a PciIO object with the (default) options digitize=False, normalize=False.")
        with open(self.io_files.conf_inp_prior, "r") as f:
            header_gen = read_conf_inp_header_lines(f)
            params = {}
            for _, p in header_gen:
                params.update(p)
            num_confs_prior = params["Nc"]

            conf_dest_arr = np.zeros((num_confs_prior, len(self.orbs)), dtype=np.uint8)

            confs_gen = read_conf_inp_confs_lines(
                    f, num_confs_prior, self.num_stat_orbs,
                    MappingProxyType(self.orbs), conf_dest_arr
                )
            for _ in confs_gen:
                pass
        return conf_dest_arr


    def _read_weights(self, flnm):
        with open(flnm, "r") as f:
            header_gen = read_conf_res_header_lines(f)
            params = {}
            for _, p in header_gen:
                params.update(p)
            num_confs = params["Nc"]
            weights_dest_arr = np.zeros(
                    (num_confs, self.num_levels)
                )
            weights_gen = read_conf_res_weights_lines(
                    f, num_confs, self.num_levels, weights_dest_arr
                )
            for _ in weights_gen:
                pass
        return self.weights_aggregator(weights_dest_arr)
            

    #override
    def read_prior_weights(self):
        return self._read_weights(self.io_files.conf_res_prior)


    #override
    def read_current_weights(self):
        return self._read_weights(self.io_files.conf_res_current)


    #override
    def write_current_basis(self, which_write):
        with open(self.io_files.conf_inp_current, "w") as g:
            with open(self.io_files.conf_inp_full) as f:
                
                header_gen = read_conf_inp_header_lines(f)
                for ln, p in header_gen:
                    if "Nc" in p:
                        ln = f" Nc ={which_write.sum()}\n"
                    g.write(ln)
                    
                confs_gen = read_conf_inp_confs_lines(
                    f, self.num_confs_full, self.num_stat_orbs
                )
                
                nonrel_indx_last = None
                something_in_nonrel = True
                rel_lns = []
        
                for ln, nonrel_indx, rel_indx in confs_gen:                    
                    if nonrel_indx is None:
                        g.write(ln)
                        continue
                        
                    if nonrel_indx != nonrel_indx_last:
                        if something_in_nonrel:
                            g.write("\n")
                        nonrel_indx_last = nonrel_indx
                        something_in_nonrel = False
                        rel_lns.clear()
        
                    rel_lns.append(ln)
                    if rel_indx is None:
                        continue
                        
                    if which_write[rel_indx]:
                        for rel_ln in rel_lns:
                            g.write(rel_ln)
                        something_in_nonrel = True
                        
                    rel_lns.clear()
