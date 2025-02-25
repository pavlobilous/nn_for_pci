import sys
sys.path.append("..")

from nn_for_pci import PciIO, PciIOFiles, NeuralManager

pci_io_files = PciIOFiles(
    conf_inp_full="full/CONF.INP",
    conf_inp_start="prior/CONF.INP",
    conf_res_start="prior/CONF.RES",
    conf_inp_current="CONF.INP",
    conf_res_current="CONF.RES"
)

pci_io = PciIO(pci_io_files)

mng = NeuralManager(pci_io)

mng.start_new_comp(0.05)
mng.save_comp("_saved_comp")
