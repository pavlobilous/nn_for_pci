from collections import namedtuple
from typing import Callable, Sequence
from types import MappingProxyType
import numpy as np

from ..neural_manager import AtomicCodeIO
from .read_files import *


weights_aggregator = lambda arr: arr.max(axis=1)

PciIOFiles = namedtuple("PciIOFiles",
            "conf_inp_full conf_inp_start conf_res_start conf_inp_current conf_res_current".split())

def _digitize_arr(arr):
    unp =  np.unpackbits(arr, axis=1)
    useless = (unp == 1).all(axis=0) | (unp == 0).all(axis=0)
    return unp[:, ~useless]


class PciIO(AtomicCodeIO):

    def __init__(self, io_files: PciIOFiles,
                 *,
                 digitize: bool = False):
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

        self.digitize = digitize
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
        if self.digitize:
            conf_dest_arr = _digitize_arr(conf_dest_arr)
        return conf_dest_arr


    #override
    def read_start_basis(self):
        if self.digitize:
            raise RuntimeError("The start basis can't be processed if the full basis features are transformed (digitized). "\
                               "Create a PciIO object with the (default) option digitize=False.")
        with open(self.io_files.conf_inp_start, "r") as f:
            header_gen = read_conf_inp_header_lines(f)
            params = {}
            for _, p in header_gen:
                params.update(p)
            num_confs_start = params["Nc"]

            conf_dest_arr = np.zeros((num_confs_start, len(self.orbs)), dtype=np.uint8)

            confs_gen = read_conf_inp_confs_lines(
                    f, num_confs_start, self.num_stat_orbs,
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
    def read_start_weights(self):
        return self._read_weights(self.io_files.conf_res_start)


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
