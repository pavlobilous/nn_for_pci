from typing import Dict, Generator, TextIO, Callable
from numbers import Real 
import re


def params_from_header_line(hdr_ln: str
                            ) -> Dict[str, Real]:
    hdr_ln = hdr_ln.split("#")[0]
    params = {}
    pairs = re.findall(r"[^\W\d_]+[\d_]*\s*=\s*\d+", hdr_ln)
    for pair in pairs:
        pair = re.sub(r"\s*=\s*", "=", pair)
        lhs, rhs = pair.split("=")
        try: 
            rhs = int(rhs)
        except ValueError:
            try:
                rhs = float(rhs)
            except ValueError:
                continue
        params[lhs] = rhs
    return params


def read_header_lines(f: TextIO,
                      stop_condition: Callable[[str], bool]
                      ) -> Generator:
    while True:
        try:
            hdr_ln = next(f)
        except StopIteration:
            raise IOError('File ended before meeting "stop_condition".')
        if stop_condition(hdr_ln):
            yield hdr_ln, {}
            return
        params = params_from_header_line(hdr_ln)
        yield hdr_ln, params


def read_conf_inp_header_lines(f: TextIO) -> Generator:
    name = next(f)
    yield name, {}
    stop_condition = lambda s: "=" not in s
    yield from read_header_lines(f, stop_condition)


def read_conf_res_header_lines(f: TextIO) -> Generator:
    stop_condition = lambda s: "====" in s
    yield from read_header_lines(f, stop_condition)
