import numpy as np
import tensorflow as tf


def train_nn(state_arrs, full_basis, weights, cutlog, nn_model, start_eval_kwargs, train_kwargs):
    cutoff = 10**cutlog

    X_train = full_basis[state_arrs["train"], :]
    y_train = tf.keras.utils.to_categorical(weights[state_arrs["train"]] > cutoff)

    iiss = np.random.permutation(X_train.shape[0])
    X_train = X_train[iiss]
    y_train = y_train[iiss]

    if start_eval_kwargs.get('verbose'):
        print("+++++ STARTING EVALUATION +++++")
    start_eval = nn_model.evaluate(X_train, y_train, **start_eval_kwargs)    

    if train_kwargs.get('verbose'):
        print("+++++ TRAINING +++++")
    train_hist=nn_model.fit(X_train, y_train, **train_kwargs)

    return start_eval, train_hist



def apply_nn(state_arrs, full_basis, nn_model, apply_kwargs):
    X_apply = full_basis[state_arrs["apply"]]

    if apply_kwargs.get('verbose'):
        print("+++++ PREDICTION +++++")
    y_apply = nn_model.predict(X_apply, **apply_kwargs)

    predimp = (y_apply[:, 1] >= 0.5)
    predimp_inds = np.where(state_arrs["apply"])[0][predimp]
    prednotimp_inds = np.where(state_arrs["apply"])[0][~predimp]

    return predimp_inds, prednotimp_inds



def balance(predimp_inds, prednotimp_inds, bal_ratio):
    num_bal_req = int(bal_ratio * predimp_inds.shape[0])
    num_bal_max = prednotimp_inds.shape[0]
    if num_bal_req < num_bal_max:
        bal_inds = np.random.choice(prednotimp_inds, num_bal_req, replace=False)
    else:
        bal_inds = prednotimp_inds.copy()

    return bal_inds



def toggle_state_arrs(state_arrs, weights, cutlog, predimp_inds, bal_inds):
    are_notimp = state_arrs["train"] & (weights <= 10**cutlog)
    state_arrs["onoff"][are_notimp] = False

    state_arrs["train"][:] = False
    state_arrs["train"][predimp_inds] = True
    state_arrs["train"][bal_inds] = True

    state_arrs["apply"][:] = ~state_arrs["onoff"]
    state_arrs["apply"][predimp_inds] = False
    state_arrs["apply"][bal_inds] = False

    state_arrs["onoff"][state_arrs["train"]] = True
