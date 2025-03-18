from typing import TextIO, Generator


def read_lines_till_weights(f: TextIO) -> Generator:
    while True:
        ln = next(f)
        yield ln
        if "ICONF" in ln:
            break
    yield next(f)


def read_conf_res_part_weights_lines(f: TextIO, Nc, dest_arr=None) -> Generator:
    if dest_arr is not None:
        assert dest_arr.shape[0] == Nc, 'dest_arr must have Nc rows.'

    for i in range(Nc):
        ln = next(f)
        yield ln
        if dest_arr is not None:
            weights =[float(v) for v in ln.split()[1:]]
            dest_arr[i, :] = weights


def read_conf_res_weights_lines(f: TextIO, Nc, Nlv, dest_arr=None) -> Generator:
    if dest_arr is not None:
        assert dest_arr.shape[0] == Nc, 'dest_arr must have Nc rows.'
        assert dest_arr.shape[1] == Nlv, 'dest_arr must have Nlv columns.'

    CHUNK_SIZE = 5
    for lv_start in range(0, Nlv, CHUNK_SIZE):
        yield from read_lines_till_weights(f)
        lv_end = min(lv_start + CHUNK_SIZE, Nlv)
        dest_part_arr = dest_arr[:, lv_start:lv_end] \
                        if dest_arr is not None \
                        else None
        yield from read_conf_res_part_weights_lines(f, Nc, dest_part_arr)


