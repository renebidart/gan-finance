# gan-finance
Generative adversarial net for financial data

A simple generative adversarial net with a 1-D convnet. Goal is to make it learn the distribution of an SDE, the Heston model.

* Was having difficulty training this properly. Need to fix this.
* One problem is that when using a CNN for this you should make the output length much larger than the length of the output vector you want.
* If a short time range is used, the model won't learn to produce large, rare price movements. It will fit to the average, and be a poor model.

