import numpy as np
import os

import torch
from torch.utils.data import TensorDataset
from ..DataGenerator import DataGenerator


class chicago_Data:
    def getData(self):
        context_val, target_val = DataGenerator(self.dataset_d,
                                        self.val_extent,
                                        self.n_x, 
                                        self.n_y,
                                        self.shift,
                                        self.zone_ids,
                                        self.map_shape,
                                        self.split,
                                        self.time_aware,
                                        self.group,
                                        ).datos
        val_data = context_val['time_series'], context_val['time'], target_val['time_series'], target_val['time']
        

        #TEST DATA
        self.split = "test"
        context_test, target_test = DataGenerator(self.dataset_d,
                                        self.test_extent,
                                        self.n_x, 
                                        self.n_y,
                                        self.shift,
                                        self.zone_ids,
                                        self.map_shape,
                                        self.split,
                                        self.time_aware,
                                        self.group
                                        ).datos
        test_data = context_test['time_series'], context_test['time'], target_test['time_series'], target_test['time']
    def _read(self, split):
        with np.load(os.path.join("./data/chicago/clean/", f"{split}.npz")) as f:
            x_ts = f["x_ts"] #train(669, 4, 90, 60), val(285, 4, 90, 60), test(381, 4, 90, 60)
            x_time = f["x_time"] #train(669, 8), val(285, 8), test(381, 8)
            y_ts = f["y_ts"] #train(669, 4, 90, 60), val(285, 4, 90, 60), test(381, 4, 90, 60)
            y_time = f["y_time"] #train(669, 8), val(285, 8), test(381, 8)

        return x_ts, x_time, y_ts, y_time
   
    def __init__(self, path):
        self.path = path
        #CONTEXT/TARdGET
        c_ts_train, c_time_train, t_ts_train, t_time_train = self._read("train") #train(9597, 4, 90, 60), train(9597, 8), train(9597, 4, 90, 60),  train(9597, 8)
        c_ts_val, c_time_val, t_ts_val, t_time_val = self._read("val") #val(2973, 4, 90, 60), val(2973, 8), val(2973, 4, 90, 60),  val(2973, 8)
        c_ts_test, c_time_test, t_ts_test, t_time_test = self._read("test") #test(4029, 4, 90, 60), test(4029, 8), test(4029, 4, 90, 60),  test(4029, 8)

        self.scale_max = c_ts_train.max((0, 1)) #valores mas grandes espaciales en el array de train context space
        c_ts_train = self.scale(c_ts_train)
        t_ts_train = self.scale(t_ts_train)

        c_ts_val = self.scale(c_ts_val)
        t_ts_val = self.scale(t_ts_val)

        c_ts_test = self.scale(c_ts_test)
        t_ts_test = self.scale(t_ts_test)

        self.train_data = (c_ts_train, c_time_train, t_ts_train, t_time_train) #{(9597, 4, 90, 60), (9597, 8), (9597, 4, 90, 60), (9597, 8)}
        self.val_data = (c_ts_val, c_time_val, t_ts_val, t_time_val) #{(2973, 4, 90, 60), (2973, 8), (2973, 4, 90, 60), (2973, 8)}
        self.test_data = (c_ts_test, c_time_test, t_ts_test, t_time_test) #{(4029, 4, 90, 60), (4029, 8), (4029, 4, 90, 60), (4029, 0)}


    def scale(self, x):
        #print("SCALE:", x / self.scale_max)
        return x / self.scale_max

    def inverse_scale(self, x):
        #print("INVERSE_SCALE:", x * self.scale_max)
        return x * self.scale_max

def chicago_Torch(data: chicago_Data, split: str):
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
    data = chicago_Data(path="data/metr_la/")
    dset = chicago_Torch(data, "test")
    breakpoint()
