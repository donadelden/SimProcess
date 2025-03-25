# VariationalRecurrentNeuralNetwork

Most of this code has been taken from [this github repository](https://github.com/emited/VariationalRecurrentNeuralNetwork).

Run `train.py` and then generate with `generate.py`. Parameters and source files should be set up at the start of both main sections. 

---

# Original Readme

Pytorch implementation of the Variational RNN (VRNN), from *A Recurrent Latent Variable Model for Sequential Data*.


The paper is available [here](https://arxiv.org/abs/1506.02216).

![png](images/fig_1_vrnn.png)

## Run:

To train: ``` python train.py ```


To sample with saved model: ``` python sample.py [saves/saved_state_dict_name.pth]```

## Some samples:

![png](images/samples.png)
