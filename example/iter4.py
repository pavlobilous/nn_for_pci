import numpy as np

import sys
sys.path.append("..")

from nn_for_pci import PciIO, PciIOFiles, NeuralManager

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

pci_io_files = PciIOFiles(
    conf_inp_full="full/CONF.INP",
    conf_inp_prior="prior/CONF.INP",
    conf_res_prior="prior/CONF.RES",
    conf_inp_current="CONF.INP",
    conf_res_current="CONF.RES"
)

pci_io = PciIO(pci_io_files, digitize=True)

mng = NeuralManager(pci_io)

mng.load_comp("_saved_comp3")
model = load_model("_saved_model3.keras")

es = EarlyStopping(
    monitor='val_accuracy',
    restore_best_weights=True,
    patience=5
)

start_eval_params = {
    'batch_size': 32 * 32 * 32,
    'verbose': 2
}

train_params = {
    'epochs': 200,
    'validation_split': 0.2,
    'verbose': 2,
    'callbacks': [es]
}

apply_params = {
    'batch_size' : 32 * 32 * 32,
    'verbose': 0
}


cutlog = -10.0
bal_frac = 0.5

mng.neural_sortout(cutlog, bal_frac, model,
            start_eval_params, train_params, apply_params)

mng.save_comp("_saved_comp4")
model.save("_saved_model4.keras")
