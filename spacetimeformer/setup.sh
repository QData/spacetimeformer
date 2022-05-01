#!/bin/sh
export STF_LOG_DIR="/home/ubuntu/spacetimeformer/spacetimeformer/log"
export STF_WANDB_ACCT="ferdinandl007"
export STF_WANDB_PROJ="spacetimeformerCrypto"
wandb login
pip install -r requirements.txt
cd ..
pip install -e .
cd spacetimeformer