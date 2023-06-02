from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import tables as tb


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    """
    Generate samples from
    :param df: .h5 file
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape #(34272, 207)
    data = np.expand_dims(df.values, axis=-1) #(34272, 207, 1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D") #(34272) --> fraccion del tiempo que ha pasado del dia
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)) #(1, 207, 34272) --> (1, num_nodes = num veces "time_ind", time_ind) --> transpose --> (34272, 207,1)
        data_list.append(time_in_day) #[(34272, 207,1), (34272, 207,1)]
       
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7)) #(34272, 207, 7) lleno de 0s
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1 #Se le pone a uno a un valor de cada fila indicando el dia de la semana que es --> lunes: [1, 0, 0, 0, 0, 0, 0]

        data_list.append(day_in_week)
    data = np.concatenate(data_list, axis=-1) #se juntan los valores de la ultima dimension (34272, 270, 1 + 1 + 7) --> [65.444, 0.00427, 0, 0, 0, 1, 0, 0, 0] = 
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    min_t = abs(min(x_offsets)) #11
    max_t = abs(num_samples - abs(max(y_offsets)))  # 34260
    for t in range(min_t, max_t): # repite 34249 veces los 12 instantes de cada momento --> 34249(num de iteraciones en el for) * (12, 207, 9) --> se multiplica en el append
        x_t = data[t + x_offsets, ...] #(12, 207, 9) --> 12 = t + x_offsets, 207 y 9 = lo mismo que hasta ahora
        y_t = data[t + y_offsets, ...] # (12, 207, 9) --> 12 = t + x_offsets, 207 y 9 = lo mismo que hasta ahora
        x.append(x_t) # añade un x_t hasta que haya 34249 --> [(12, 207, 9)] --> [(12, 207, 9), (12, 207, 9)] --> ... --> [(12, 207, 9), (12, 207, 9), ..., (12, 207, 9)] 
        y.append(y_t) # añade un y_t hasta que haya 34249 --> [(12, 207, 9)] --> [(12, 207, 9), (12, 207, 9)] --> ... --> [(12, 207, 9), (12, 207, 9), ..., (12, 207, 9)]

    x = np.stack(x, axis=0) # (34249, 12, 207, 9) --> añade todo el x.append(x_t) en la primera dimension creando un 4D Array
    y = np.stack(y, axis=0) # (34249, 12, 207, 9) --> añade todo el y.append(y_t) en la primera dimension creando un 4D Array

    return x, y


def generate_train_val_test():
    df = pd.read_hdf("data/metr_la/metr-la.h5")
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),)) # [-11 -10  -9  -8  -7  -6  -5  -4  -3  -2  -1   0]
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1)) # [ 1  2  3  4  5  6  7  8  9 10 11 12]
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,
    )
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join("data/metr_la", "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )




def main():
    print("Generating training data")
    generate_train_val_test()


if __name__ == "__main__":
    main()
