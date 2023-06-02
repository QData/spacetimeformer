import numpy as np
import os

import torch
from torch.utils.data import TensorDataset


class METR_LA_Data:
    def _read(self, split):
        with np.load(os.path.join(self.path, f"{split}.npz")) as f:
            x = f["x"] #train(23974, 12, 207, 9), val(3425, 12, 207, 9), test(6850, 12, 207, 9)
            y = f["y"] #train(23974, 12, 207, 9), val(3425, 12, 207, 9), test(6850, 12, 207, 9)
        return x, y
    def _split_set(self, data):
        # time features are the same across the 207 nodes.
        # just grab the first one
        #print("data:", data.shape)
        x = np.squeeze(data[:, :, 0, 1:]) #tercera dimension = 0 porque cada 207 tiene el mismo valor de tiempo y se cogen todos los valores
                                          #de tiempo de la ultima dimension: time_of_day + day_of_week (el primer valor es df.values)
                                          #train(23974, 12, 8), val(3425, 12, 8), test(6850, 12, 8)


        #normalizar el tiempo que ha pasado del dia 
        time = 2.0 * x[:, :, 0] - 1.0 #train(23974, 12), val(3425, 12), test(6850, 12)
        # convert one-hot day of week feature to scalar [-1, 1]
        day_of_week = x[:, :, 1:] #se coge los [0, 0, 1, 0, 0, 0, 0] --> #train(23974, 12, 7), val(3425, 12, 7), test(6850, 12, 7)
        day_of_week = np.argmax(day_of_week, axis=-1).astype(np.float32) #obtener el indice de donde esta el 1(dia de la semana) --> #train(23974, 12)
        day_of_week = 2.0 * (day_of_week / 6.0) - 1.0 #normalizar el indice del dia de la semana
        time = np.expand_dims(time, axis=-1) #se añade el tiempo normalizado en la ultima dimension --> #train(23974, 12, 1), val(3425, 12, 1), test(6850, 12, 1)


        day_of_week = np.expand_dims(day_of_week, axis=-1) #lo añade en la en la ultima dimension 
        x = np.concatenate((time, day_of_week), axis=-1) #train(23974, 12, 1 + 1), val(3425, 12, 1 + 1), test(6850, 12, 1 + 1)
        y = data[:, :, :, 0] #solo los df.values de la ultima dimension por lo que pasa a --> train(23974, 12, 207), val(3425, 12, 207), test(6850, 12, 207)


        return x, y

    def __init__(self, path):
        self.path = path
        #CONTEXT/TARGET
        context_train, target_train = self._read("train") #train(23974, 12, 207, 9), train(23974, 12, 207, 9)
        context_val, target_val = self._read("val") #val(3425, 12, 207, 9), val(3425, 12, 207, 9)
        context_test, target_test = self._read("test") #test(6850, 12, 207, 9), test(6850, 12, 207, 9)
        

        #X_Y CONTEXT/X_Y TARGET
        x_c_train, y_c_train = self._split_set(context_train) #x_c_train(23974, 12, 2) , #y_c_train(23974, 12, 207)
        x_t_train, y_t_train = self._split_set(target_train) #x_t_train(23974, 12, 2) , #y_t_train(23974, 12, 207)

        x_c_val, y_c_val = self._split_set(context_val) #x_c_val(3425, 12, 2), y_c_val(3425, 12, 207)
        x_t_val, y_t_val = self._split_set(target_val) #x_t_val(3425, 12, 2), y_t_val(3425, 12, 207)

        x_c_test, y_c_test = self._split_set(context_test) #x_c_test(6850, 12, 2), y_c_test(6850, 12, 207)
        x_t_test, y_t_test = self._split_set(target_test) #x_c_test(6850, 12, 2), y_c_test(6850, 12, 207)


        self.scale_max = y_c_train.max((0, 1)) #valores mas grandes espaciales en un array de (207)
        y_c_train = self.scale(y_c_train)
        y_t_train = self.scale(y_t_train)

        y_c_val = self.scale(y_c_val)
        y_t_val = self.scale(y_t_val)

        y_c_test = self.scale(y_c_test)
        y_t_test = self.scale(y_t_test)

        self.train_data = (x_c_train, y_c_train, x_t_train, y_t_train) #{(23974, 12, 2), (23974, 12, 207), (23974, 12, 2, (23974, 12, 207)}
        self.val_data = (x_c_val, y_c_val, x_t_val, y_t_val) #{(3425, 12, 2), (3425, 12, 207), (3425, 12, 2), (3425, 12, 207)}
        self.test_data = (x_c_test, y_c_test, x_t_test, y_t_test) #{(6850, 12, 2), (6850, 12, 207), (6850, 12, 2), (6850, 12, 207)}


    def scale(self, x):
        #print("SCALE:", x / self.scale_max)
        return x / self.scale_max

    def inverse_scale(self, x):
        #print("INVERSE_SCALE:", x * self.scale_max)
        return x * self.scale_max

def METR_LA_Torch(data: METR_LA_Data, split: str):
    assert split in ["train", "val", "test"]
    if split == "train":
        print("pasa por train")
        tensors = data.train_data
    elif split == "val":
        print("pasa por val")
        tensors = data.val_data
    else:
        print("pasa por test")
        tensors = data.test_data
    tensors = [torch.from_numpy(x).float() for x in tensors]
    return TensorDataset(*tensors) #el modelo utiliza tensores de torch


if __name__ == "__main__":
    data = METR_LA_Data(path="data/metr_la/")
    dset = METR_LA_Torch(data, "test")
    breakpoint()
