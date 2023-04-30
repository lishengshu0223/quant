import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
import pandas as pd

# load data
data = pd.read_pickle('../data/factor_stack_2019_01.pkl')
target = pd.read_pickle('../data/quantile_return_2019_01.pkl')
factor_names = list(data.columns)
data['target'] = target     # target列为需要预测的label
data.index.names = ['date', 'code']
data = data.reset_index()
unique_date = pd.unique(data['date'])
date2idx = {d: i for i, d in enumerate(unique_date)}
data['time_idx'] = data['date'].apply(lambda d: date2idx[d.to_datetime64()])

gb = data.groupby('code')
valid_code = []
for code, labels in gb.groups.items():
    if len(labels) >= 20:
        valid_code.append(code)
data = data[data['code'].isin(valid_code)]

training_cutoff = data["time_idx"].max() - 5
max_encoder_length = int(training_cutoff - 3)
max_prediction_length = 3
# print(f"{max_encoder_length=}, {max_prediction_length=}")
# print(f"{type(max_encoder_length)=}, {type(max_prediction_length)=}")

training = TimeSeriesDataSet(
    data[lambda x: x['time_idx'] < training_cutoff],
    time_idx='time_idx',
    target='target',
    group_ids=['code'],
    max_encoder_length=max_encoder_length,
    min_encoder_length=max_encoder_length // 2,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=['code'],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=['time_idx'],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=factor_names,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# create validation and training dataset
validation = TimeSeriesDataSet.from_dataset(training, data,
                                            min_prediction_idx=training.index.time.max() + 1,
                                            stop_randomization=True,
                                            )
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

# define trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)

# create the model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=2,
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
res = Tuner(trainer).lr_find(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model
trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)

preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)

print(f"{preds=}, {index=}")
