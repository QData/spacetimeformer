from argparse import ArgumentParser
import random
import sys
import warnings
import os
import uuid

import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
import datetime as dt
import pandas as pd
import tables as tb
import numpy as np



import data
import callbacks
import mdst_transformer_model
import plot
import wandb

class mdst_transformer():
    def __init__(self, work_path= '.', time_gran='15m', city='chicago',
                 kind='map_90_60', dset_conv='norm_abs',
                 n_x=4, n_y=4, shift=1,
                 batch_size=2**8, time_aware=True,
                 zone_ids=slice(None),
                 ):
        
        self.run_name = 'MDST_Transformer'
        self.no_earlystopping = False
        self.time_mask_loss = True

        # For files and paths
        self.work_path = work_path
        self.time_gran = time_gran

        self.path = os.path.join(self.work_path, 'data','metr_la')
        self.models_path = os.path.join(self.work_path, 'models', city, 'shifted')
        self.data_path = os.path.join(self.work_path, 'data', city, 'clean')
        self.other_path = os.path.join(self.work_path, 'data', city, 'other')

        self.dataset = '{}_{}_{}_{}.h5'
        self.dataset_d = {'time_series':[os.path.join(self.data_path, self.dataset.format(time_gran, kind, 'taxi', dset_conv))]}

        self.group = '/2013'

        if time_aware:
            self.dataset_d['holidays'] = [os.path.join(self.other_path, 'holidays.csv')]
        self.city, self.kind, self.dset_conv = city, kind, dset_conv

        # For training and prediction
        self.n_x = n_x
        self.n_y = n_y
        self.shift = shift
        self.batch_size = batch_size
        self.time_aware = time_aware

        # Train limits
        ints_in_day = 4 * 24 # intervals in a day (depends on time_gran)
        train_lim = 7 * ints_in_day #max index for training data ((7 days * num intervals per day)
        self.train_extent = (0, train_lim) #(0, 672)
        print("self.train_extent", self.train_extent) #672

        # Validation limits
        val_lim = train_lim + 1 * 3 * ints_in_day #max index for validation data (train_lim + (intervals in 3 days))
        self.val_extent = (train_lim, val_lim)#(672, 960)
        print("self.val_extent", self.val_extent) #288

        #Test limits
        test_lim = self.__datetime_to_idx(dt.datetime(2020, 3, 16, 0, 0)) #test limit set to begining of COVID pandemic idx = 252576
        self.test_extent = (val_lim, test_lim) #(960, 1344)
        print("self.test_extent",self.test_extent)# 384

        self.df_stats = pd.read_csv(os.path.join(self.other_path, 'metrics-per-zone-taxi.csv'), index_col=0) #id de las zonas taxi

        with tb.open_file('./data/chicago/clean/15m_flat_taxi_count.h5', mode='r') as h5_taxi:
            t = h5_taxi.get_node('/2013')[:].mean(axis=0)
        self.tnorm = (t - t.min()) / (t.max() - t.min()) 

        # Target variable
        self.target_d = {'flat_count': 'Trip counts',
                         'map_40_40_count': 'Trip counts map 40x40 [-]',
                         'map_90_60_count': 'Trip counts map 90x60 [-]',
                         'flat_norm': 'Trip counts normalized per zone',
                         'map_40_40_norm': 'Normalized trip counts map 40x40 [-]',
                         'map_90_60_norm': 'Normalized trip counts map 90x60 [-]',
                         'flat_norm_abs': 'Trip counts normalized abs.',
                         'map_40_40_norm_abs': 'Normalized abs. trip counts map 40x40 [-]',
                         'map_90_60_norm_abs': 'Normalized abs. trip counts map 90x60 [-]',
                         'norm_abs_050': 'Trip counts normalized abs. at 50',
                         'flat_stand': 'Trip counts standardized per zone',
                         'map_40_40_stand': 'Standardized trip counts map 40x40 [-]',
                         'flat_stand_abs': 'Trip counts standardized abs.',
                         'map_40_40_stand_abs': 'Standardized abs. trip counts map 40x40 [-]'
                        }
        self.target = self.target_d['{}_{}'.format(kind, dset_conv)]
        with tb.open_file(self.dataset_d['time_series'][0], mode='r') as h5_file:
            self.zones = h5_file.get_node(self.group)._v_attrs['columns']
        self.zone_ids = zone_ids


        self.set_kind(self.kind)

    def generatingDataset(self):
        #TRAIN DATA
        self.split = "train"
        context_train, target_train = data.DataGenerator(self.dataset_d,
                                        self.train_extent,
                                        self.n_x, 
                                        self.n_y,
                                        self.shift,
                                        self.zone_ids,
                                        self.map_shape,
                                        self.split,
                                        self.time_aware,
                                        self.group,
                                        ).datos
        train_data = (context_train['time_series'], context_train['time'], target_train['time_series'], target_train['time'])

        #VALIDATION DATA
        self.split = "val"
        context_val, target_val = data.DataGenerator(self.dataset_d,
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
        context_test, target_test = data.DataGenerator(self.dataset_d,
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
        datos = (train_data, val_data, test_data)
        return datos
    
    def chicago_Torch(self, split: str):
        assert split in ["train", "val", "test"]
        datos = self.generatingDataset()
        if split == "train":
            tensors = datos[0]
        elif split == "val":
            tensors = datos[1]
        else:
            tensors = datos[2]
        tensors = [torch.from_numpy(x).float() for x in tensors]
        return TensorDataset(*tensors) #el modelo utiliza tensores de torch


    # Map shape for convolutionals
    def get_kind(self):
        return self.kind
    
    def set_kind(self, x): #CREACION DEL MESH-GRID
        if 'map' in x:
            # Load longitude and latitude for the interpolation
            st_path = os.path.join(self.other_path, 'zone-centroids-taxi.csv')
            self.xy_taxi = pd.read_csv(st_path).loc[:, ['longitude', 'latitude']].values
            self.lng_taxi = self.xy_taxi[:, 0]
            self.lat_taxi = self.xy_taxi[:, 1]
            st_path = os.path.join(self.other_path, 'station-locations-bike.csv')
            self.xy_bike = pd.read_csv(st_path).loc[:, ['lng', 'lat']].values
            self.lng_bike = self.xy_bike[:, 0]
            self.lat_bike = self.xy_bike[:, 1]
            st_path = os.path.join(self.other_path, 'grid-locations-bike.csv')
            self.xy_bike_g = pd.read_csv(st_path).loc[:, ['lng_int', 'lat_int']].values
            self.lng_bike_g = self.xy_bike_g[:, 0]
            self.lat_bike_g = self.xy_bike_g[:, 1]
            self.map_shape = tuple([int(aux) for aux in self.kind.split('_')[-2:]])
            offset = 0.002
            xrange = (min(self.lng_taxi.min(), self.lng_bike.min()), max(self.lng_taxi.max(), self.lng_bike.max()))
            yrange = (min(self.lat_taxi.min(), self.lat_bike.min()), max(self.lat_taxi.max(), self.lat_bike.max()))
            xstep = (xrange[1] - xrange[0] + 2 * offset) / self.map_shape[1]
            ystep = (yrange[1] - yrange[0] + 2 * offset) / self.map_shape[0]
            xnew = np.linspace(xrange[0] - offset + xstep/2, xrange[1] + offset - xstep/2, self.map_shape[1])
            ynew = np.linspace(yrange[0] - offset + ystep/2, yrange[1] + offset - ystep/2, self.map_shape[0])
            self.X, self.Y = np.meshgrid(xnew, ynew, indexing='ij')
            self.grid = np.stack((self.X.ravel(), self.Y.ravel())).T
        else:
            self.zone_ids = np.arange(801)
            self.map_shape = len(self.zone_ids)
        #self.model.set_map_shape(self.map_shape)

    def create_run_name(self, add):
        self.run_name = self.run_name + "_" + add
        return self.run_name
    
    def __idx_to_datetime(self, idx, year=2013, freq=None):
    # Frequency expressed in minutes
        if freq is None:
            freq = int(self.time_gran.replace('m', ''))
        base = dt.datetime(year, 1, 1, 0, 0)
        return base + dt.timedelta(minutes=freq * idx)

    def __datetime_to_idx(self, date, freq=None):
        # Frequency expressed in minutes
        if freq is None:
            freq = int(self.time_gran.replace('m', ''))
        base = dt.datetime(2020, 3, 2, 0, 0)
        return int((date - base).total_seconds() / (60 * freq))


    def create_model(self):
        x_dim = 8
        yc_dim = 60
        yt_dim = 60
        assert x_dim is not None
        assert yc_dim is not None
        assert yt_dim is not None

        if hasattr(self, "n_x") and hasattr(self, "n_y"):
            max_seq_len = self.n_x + self.n_y
        elif hasattr(self, "max_len"):
            max_seq_len = None
        else:
            raise ValueError("Undefined max_seq_len")
        
        forecaster = mdst_transformer_model.mdst_transformer_forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            max_seq_len=max_seq_len,
            start_token_len=64,
            attn_factor=5,
            d_model=200,
            d_queries_keys=50,
            d_values=50,
            n_heads=4,
            e_layers=2,
            d_layers=2,
            d_ff=800,
            dropout_emb=0.1,
            dropout_attn_out=0.0,
            dropout_attn_matrix=0.0,
            dropout_qkv=0.0,
            dropout_ff=0.2,
            pos_emb_type="abs",
            use_final_norm="performer",
            global_self_attn="performer",
            local_self_attn="performer",
            global_cross_attn="performer",
            local_cross_attn="performer",
            performer_kernel="relu",
            performer_redraw_interval=1000,
            attn_time_windows=1,
            use_shifted_time_windows=True,
            norm="batch",
            activation="gelu",
            init_lr=1e-10,
            base_lr=5e-4,
            warmup_steps=1000,
            decay_factor=0.8,
            initial_downsample_convs=0,
            intermediate_downsample_convs=0,
            embed_method="spatio-temporal",
            l2_coeff=1e-3,
            loss="mse",
            class_loss_imp=1e-3,
            recon_loss_imp=0,
            time_emb_dim=6,
            null_value=None,
            pad_value=None,
            linear_window=0,
            use_revin=False,
            linear_shared_weights=False,
            use_seasonal_decomp=False,
            use_val=True,
            use_time=True,
            use_space=True,
            use_given=True,
            recon_mask_skip_all=1.0,
            recon_mask_max_seq_len=5,
            recon_mask_drop_seq=0.1,
            recon_mask_drop_standard=0.2,
            recon_mask_drop_full=0.05,
            )
        return forecaster

    def create_dset(self):
        NULL_VAL = None
        PLOT_VAR_IDXS = None
        PLOT_VAR_NAMES = None
        PAD_VAL = None
        datos = self.generatingDataset()
        DATA_MODULE = data.DataModule(
            datasetCls= self.chicago_Torch,
            dataset_kwargs= datos,
            batch_size=self.batch_size,
            workers=4,
            overfit=False,
        )
        NULL_VAL = 0.0

        return (
            DATA_MODULE,
            NULL_VAL,
            PLOT_VAR_IDXS,
            PLOT_VAR_NAMES,
            PAD_VAL,
        )

    def create_callbacks(self, save_dir):

        filename = f"{self.run_name}_" + str(uuid.uuid1()).split("-")[0]
        model_ckpt_dir = os.path.join(save_dir, filename)
        self.model_ckpt_dir = model_ckpt_dir
        saving = pl.callbacks.ModelCheckpoint(
            dirpath=model_ckpt_dir,
            monitor="val/loss",
            mode="min",
            filename=f"{self.run_name}" + "{epoch:02d}",
            save_top_k=1,
            auto_insert_metric_name=True,
        )
        llamadas = [saving]
        
        if not self.no_earlystopping:
            llamadas.append(
                pl.callbacks.early_stopping.EarlyStopping(
                    monitor="val/loss",
                )
            )

        llamadas.append(pl.callbacks.LearningRateMonitor())

        if self.time_mask_loss:
            llamadas.append(
                callbacks.TimeMaskedLossCallback(start=1,
                end=12,
                steps=1000)
            )
        return llamadas


    def main(self, ind):
        ind += 1
        print("ind", ind)
        log_dir = os.getenv("wandb")
        if log_dir is None:
            log_dir = "./output/wandb"
            print(
                "Using default wandb log dir path of ./data/STF_LOG_DIR. This can be adjusted with the environment variable `STF_LOG_DIR`"
            )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
           
        project = "TFM"
        entity = "koldo-moya"
        assert (
            project is not None and entity is not None
        ), "Please set environment variables `STF_WANDB_ACCT` and `STF_WANDB_PROJ` with \n\
            your wandb user/organization name and project title, respectively."
        

        experiment = wandb.init(
            project=project,
            entity=entity,
            dir=log_dir,
            reinit=True,
        )
        config = wandb.config
        wandb.run.name = self.run_name
        wandb.run.save()
        logger = pl.loggers.WandbLogger(
            experiment=experiment,
            save_dir=log_dir,
        )

        #DATASET
        (
            data_module,
            null_val,
            plot_var_idxs,
            plot_var_names,
            pad_val,
        ) = self.create_dset()
        print("pasa dataset")
        #MODEL
        self.null_value = null_val
        self.pad_value = pad_val
        forecaster = self.create_model()
        forecaster.set_null_value(null_val)
        print("pasa modelo")
        # Callbacks
        llamadas = self.create_callbacks(save_dir=log_dir)

        test_samples = next(iter(data_module.test_dataloader()))

        llamadas.append(
            plot.PredictionPlotterCallback(
                test_samples,
                var_idxs=plot_var_idxs,
                var_names=plot_var_names,
                pad_val=pad_val,
                total_samples=min(8, self.batch_size),
            )
        )

        llamadas.append(
            plot.AttentionMatrixCallback(
                test_samples,
                layer=0,
                total_samples=min(16, self.batch_size),
            )
        )

        logger.log_hyperparams(config)

        val_control = {"val_check_interval": 1.0}

        trainer = pl.Trainer(
            callbacks=llamadas,
            logger=logger,
            accelerator="auto",
            gradient_clip_val= None,
            gradient_clip_algorithm="norm",
            overfit_batches=20,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            limit_val_batches=1,
            **val_control,
        )
         # Train
        trainer.fit(forecaster, datamodule=data_module)

        # Test
        trainer.test(datamodule=data_module, ckpt_path=b"est")

        # Predict (only here as a demo and test)
        forecaster.to("cuda")
        xc, yc, xt, _ = test_samples
        yt_pred = forecaster.predict(xc, yc, xt)
        print(yt_pred)

        experiment.finish()

if __name__ == '__main__':   
    transformer = mdst_transformer()
    #print("model", transformer.model)
    ind = 0
    transformer.main(ind)