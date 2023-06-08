import numpy as np
import pandas as pd
import tables as tb
import datetime as dt
import time, os
import torch
import matplotlib.pyplot as plt 

class DataGenerator():
        
    def __init__(self, dataset_d, extent, n_x, n_y, shift, target_ids, 
                 mesh_grid, split, time_aware=False, group='/2013'):

        # Dataset config
        self.time_series = dataset_d['time_series']
        self.holidays = dataset_d['holidays'] if 'holidays' in dataset_d.keys() else None
        if self.holidays is not None:
            self.holidays = pd.read_csv(self.holidays[0], parse_dates=[1])
        self.group = group
        self.time_aware = time_aware

        # Train/val/test extend, n_instants and split
        self.start_id, self.end_id = extent
        self.n_instants = self.end_id - self.start_id
        self.split = split
        # Profundidad x y, espacio entre x y
        self.n_x = n_x
        self.n_y = n_y
        self.shift = shift

        # MESH-GRID and target_ids
        self.x_shape = mesh_grid
        self.y_shape = mesh_grid
        self.target_ids = target_ids

        #FUNCTIONS
        self.show("Generating %s data" % self.split)
        self.datos = self.generateData()
        #self.createNPZ(self.split)

    def show(self, t):
        print()
        print('\t', t)
        print('\t     Time_series dataset: ', self.time_series)
        print('\t     Dataset group: ', self.group)
        print('\t     Time_aware: ', self.time_aware)
        print('\t     n_instants', self.n_instants)
        print('\t     start, end', self.start_id, self.end_id)
        print('\t     n_x: ', self.n_x)
        print('\t     n_y: ', self.n_y)
        print('\t     shift: ', self.shift)
        print('\t     x_shape: ', self.x_shape)
        print('\t     y_shape: ', self.y_shape)
        print('\t     target_ids: ', self.target_ids)
        print()

    def generateData(self):
        x_extent = (self.start_id, self.end_id) # (0, train/val/test) num instantes en train/val/test

        #Time series
        X_ts = np.empty((self.n_instants, self.n_x, self.x_shape[0], self.x_shape[1])) #(train/val/test, 4, 90, 60) 
        y_ts = np.empty((self.n_instants, self.n_y, self.y_shape[0], self.y_shape[1])) #(train/val/test, 4, 90, 60)

        X_ts, y_ts = self.__data_generation(self.time_series[0], *x_extent) #(train/val/test, 4, 90, 60)
        X = {'time_series': X_ts}
        Y = {'time_series': y_ts}


        # Time
        if self.time_aware:
            X['time'], Y['time']= self.__get_input_times(*x_extent)

        print("X_%s_time_series: " % self.split, X['time_series'].shape)
        print("X_%s_time: " % self.split, X['time'].shape)
        print("Y_%s_time_series: " % self.split, Y['time_series'].shape)
        print("Y_%s_time: " % self.split, Y['time'].shape)
        return X, Y


    def __idx_to_datetime(self, idx, freq, year=2013):
        # Frequency expressed in minutes
        freq = int(freq.replace('min', ''))
        base = dt.datetime(year, 1, 1, 0, 0)
        return base + dt.timedelta(minutes=int(freq * idx))

    def __get_input_times(self, x_l, x_r, freq='15min', mapper=dict(zip(range(7), [0]*5 + [1]*2))):
        '''
        Implemented to include:
        - time of day in [-1, 1],
        - time of week in [-1, 1],
        - time of year in [-1, 1],
        - week or weekend in {0, 1},
        - holiday in {0, 1}.
        '''   
        x_times = pd.date_range(start=self.__idx_to_datetime(x_l, freq),
                              end=self.__idx_to_datetime(x_r-4, freq), freq=freq)
        # Convert to seconds since epoch
        x_times_s = x_times.astype('int64') // 1e9
        day = 24 * 60 * 60
        week = 7 * day
        year = 365.2425 * day
        x_time_ret = np.stack([np.sin(x_times_s * 2 * np.pi / day), #seno de la hora del dia en un periodo de 24 horas pasadas a segundos
                             np.cos(x_times_s * 2 * np.pi / day), #coseno de la hora del dia en un periodo de 24 horas pasadas a segundos
                             np.sin(x_times_s * 2 * np.pi / week), #seno del dia de la semana en un periodo de 7 dias pasados a segundos
                             np.cos(x_times_s * 2 * np.pi / week), #cosenos del dia de la semana en un periodo de 7 dias pasados a segundos
                             np.sin(x_times_s * 2 * np.pi / year), #senos del dia del año en un periodo de 1 año pasado a segundos
                             np.cos(x_times_s * 2 * np.pi / year), #cosenos del dia del año en un periodo de 1 año pasado a segundos
                             x_times.weekday.map(mapper), #si es dia de la semana (0) o fin de semana (1)
                             x_times.normalize().isin(self.holidays.Date).astype(int)], #si son vacaciones (1) o no (0)
                            axis=1) #(train/val/test, 8)

        x_time = np.stack([x_time_ret[i:i + self.n_x] for i in range(self.n_instants-6)]) #(train/val/test, 4, 8)



        y_l, y_r = x_l + self.n_x + self.shift - 1, x_r + self.n_y + self.shift - 1

        y_times = pd.date_range(start=self.__idx_to_datetime(y_l, freq),
                              end=self.__idx_to_datetime(y_r-4, freq), freq=freq)
        # Convert to seconds since epoch
        y_times_s = y_times.astype('int64') // 1e9
        day = 24 * 60 * 60
        week = 7 * day
        year = 365.2425 * day
        y_time_ret = np.stack([np.sin(y_times_s * 2 * np.pi / day), #seno de la hora del dia en un periodo de 24 horas pasadas a segundos
                             np.cos(y_times_s * 2 * np.pi / day), #coseno de la hora del dia en un periodo de 24 horas pasadas a segundos
                             np.sin(y_times_s * 2 * np.pi / week), #seno del dia de la semana en un periodo de 7 dias pasados a segundos
                             np.cos(y_times_s * 2 * np.pi / week), #cosenos del dia de la semana en un periodo de 7 dias pasados a segundos
                             np.sin(y_times_s * 2 * np.pi / year), #senos del dia del año en un periodo de 1 año pasado a segundos
                             np.cos(y_times_s * 2 * np.pi / year), #cosenos del dia del año en un periodo de 1 año pasado a segundos
                             y_times.weekday.map(mapper), #si es dia de la semana (0) o fin de semana (1)
                             y_times.normalize().isin(self.holidays.Date).astype(int)], #si son vacaciones (1) o no (0)
                            axis=1)
        y_time = np.stack([y_time_ret[i:i + self.n_y] for i in range(self.n_instants-6)]) #(train/val/test, 4, 8)
        return x_time, y_time

    def __data_generation(self, dset_path, x_l, x_r, n_repeat=1):
        with tb.open_file(dset_path, mode='r') as h5_file:
            x_slc = h5_file.get_node(self.group)[x_l:x_r, :20, :20]  #(train/val/test, 90, 60)
            x_slc = np.repeat(x_slc, n_repeat, axis=0) #(train/val/test, 90, 60)
            X = np.stack([x_slc[i:i + self.n_x] for i in range(self.n_instants-6)]) #(train/val/test, 4, 90, 60)

            y_l, y_r = x_l + self.n_x + self.shift - 1, x_r + self.n_y + self.shift - 1
            y_slc = h5_file.get_node(self.group)[y_l:y_r, :20, :20] #(train/val/test, 90, 60)
            Y = np.stack([y_slc[i:i + self.n_y] for i in range(self.n_instants-6)]) #(train/val/test, 4, 90, 60)
        return X, Y
"""
    def createNPZ(self, split):
        x, y = self.generateData()
        np.savez_compressed(
            os.path.join("data/chicago/clean", "%s.npz" % split),
            x_ts=x['time_series'],
            x_time=x['time'],
            y_ts=y['time_series'],
            y_time=y['time']
            )
"""