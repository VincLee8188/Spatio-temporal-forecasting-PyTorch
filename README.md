# Spatio-temporal-forecasting
Leverage on recent advances in graph convolution and sequence modeling to design neural networks for spatio-temporal forecasting, which including the use of graph convolutional neural networks, gated recurrent units, encoder-decoder framework, attention mechanism and transformers.

There are two models I designed for spatio-temporal forecasting in this repository, both adapts an encoder-decoder architecture. 

The first one is devised on the basis of Diffusion Convolutional Recurrent Neural Network(DCRNN) proposed in [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926) by Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu. I added attention mechanism to the original model, which models the direct relationships between historical and future time steps, helping to alleviate the error propagation problem among prediction time steps. In addition, I implemented some regularization methods using python, such as early stop and hold out dataset, to avoid overfitting. The code of this model is included in layer_module.py and base_model.py.

In the second model, it captures the spatial dependency using graph convolutional neural network, and the temporal dependency using transformer. The code of this model is included in `transformer.py`.
