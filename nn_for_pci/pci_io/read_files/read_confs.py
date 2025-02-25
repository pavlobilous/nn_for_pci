import math
from itertools import islice, count, repeat
from typing import TextIO, Generator


def read_core_lines(f: TextIO, Nso: int) -> Generator:
    if Nso > 0:
        yield from islice(f, math.ceil(Nso / 6) - 1)


def outside_nonrel_group(ln: str) -> bool:
    return (len(ln) <= 4) or not(ln[4:].strip())


def is_new_rel_config(ln: str) -> bool:
    return bool(ln[:4].strip())


def read_line_from_nonrel_group(f: TextIO) -> str:
    try:
        ln = next(f)
    except StopIteration:
        ln = ""
    ln = ln[:66]
    if not ln.endswith("\n"):
        ln += "\n"
    return ln
    

class EofInsideGenerator(Exception):
    pass


def read_lines_till_nonrel_group(f: TextIO) -> Generator:
    while True:
        try:
            ln = read_line_from_nonrel_group(f)
        except StopIteration:
            raise EofInsideGenerator 
        yield ln
        if not outside_nonrel_group(ln):
            return


def ln_to_conf(ln: str) -> str:
    return ln[4:]
    

def read_nonrel_group_lines(f: TextIO) -> Generator:
    rltng = read_lines_till_nonrel_group(f)
    first = True
    while True:
        try:
            ln = next(rltng)
        except StopIteration:
            break
        except EofInsideGenerator:
            return
        if not first:
            yield ln_old, None
        first = False
        ln_old = ln

    first = True
    while True:
        if is_new_rel_config(ln):
            if not first:
                yield ln_old, rel_conf
            rel_conf = ln_to_conf(ln)
        else:
            yield ln_old, None
            rel_conf += " " * 4 + ln_to_conf(ln)
        first = False
        ln_old = ln
        ln = read_line_from_nonrel_group(f)
        if outside_nonrel_group(ln):
            yield ln_old, rel_conf
            yield ln, None
            break


def extract_orbs_from_conf(rel_conf: str, orb_dict: dict) -> Generator:
    chunk_orb = lambda orb: (
            orb[:-2], int(orb[-2:])
        )
    labels_and_pops = map(chunk_orb, rel_conf.split())
    for label, pop in labels_and_pops:
        if label not in orb_dict:
            orb_dict[label] = len(orb_dict)
        col = orb_dict[label]
        yield col, pop


def read_conf_inp_confs_lines(f: TextIO, Nc, Nso,
                              orb_dict=None, conf_dest_arr=None, nonrel_dest_arr=None):
    if conf_dest_arr is not None:
        assert conf_dest_arr.shape[0] == Nc, 'conf_dest_arr must have Nc rows.'
    if nonrel_dest_arr is not None:
        assert nonrel_dest_arr.shape[0] == Nc, 'nonrel_dest_arr must have Nc rows.'

    for ln in read_core_lines(f, Nso):
        yield ln, None, None

    nonrel_ctr = count()
    rel_ctr = count()
    while True:
        nonrel_indx = next(nonrel_ctr)
        for ln, rel_conf in read_nonrel_group_lines(f):
            if rel_conf is not None:
                rel_indx = next(rel_ctr)
                if orb_dict is not None:
                    cols_and_pops_gen = extract_orbs_from_conf(rel_conf, orb_dict)
                    cols, pops = zip(*cols_and_pops_gen)
                if conf_dest_arr is not None:
                    conf_dest_arr[rel_indx, cols] = pops
                if nonrel_dest_arr is not None:
                    nonrel_dest_arr[rel_indx] = nonrel_indx
                yield ln, nonrel_indx, rel_indx
                if rel_indx >= Nc - 1:
                    return
            else:
                yield ln, nonrel_indx, None 
