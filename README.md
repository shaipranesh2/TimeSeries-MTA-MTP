# TimeSeries-MTA-MTP
This repo implements multi token attention and multi token prediction for Timer Model (Time series transformer)

# Steps to recreate resutls:
1. Choose what model you want to train MTA(Multi-Token Attention), MTP(Multi-Token Prediction), MTA_MTP (MTA & MTP together), Timer (Timer transformer architecture)
2. Place your dataset in your dataset folder in those folder (UTSD) for pretraining
3. Run train67M.sh or train33M.sh for pretraining
4. Run adapt67M.sh adapt33.sh for ETTh1 finetuning.
