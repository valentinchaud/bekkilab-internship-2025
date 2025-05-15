# Survey about loss functions

## Negative Loss Likelihood

The Negative Loss Likelihood quantifies the difference between the probabilities predicted by a model and the actual 
values observed in a dataset. The idea is to penalize a model based by how unlikely it considers the true data to be.

The formula is 
![NLLFormula.png](NLLFormula.png)

Where p(x_i|θ) represents the probability/likelihood of observing the data point x_i given the model parameters θ.

If the models assigns a high probability to the true value as the observed value x_i, then p(x_i|θ) is close to 1. The 
log will then be close to 0, and then the NLL will be a small positive number. If the probability of the true value is low, the
log will be a higher negative number, thus the NLL will be a high positive number.

The main use cases for NLL are for classification, for calculating the probability of positive class (in binary 
classification) or the multiple different classes (for multiclass classification). It's also used in generative models, 
which are maximizing the likelihood of the data, which is equivalent of minimizing the NLL.

## Cross-entropy loss

Entropy has been defined by Claude Shannon as : 

![ShannonEntropy.png](ShannonEntropy.png)

This can be understood as the uncertainty of having a specific choice in a distribution of probabilities. For example, 
if we have a probability in a distribution that is close to 1, we are almost sure that we will have this choice in the 
selection, so the uncertainty is low, thus the entropy value is low.

The cross-entropy loss is a metric that measures the distance between two probabilities distribution. In the case of
classification, it's the difference between the true distribution (the true datas) and the predicted datas given by the
model. The formula is : 
![CrossEntropy.png](CrossEntropy.png)

where P(x) is the true probability of the event x, and Q(x) the probability of the predicted event x.

A lower cross-entropy value signifies a better model performance, because the predicted values are close to the true 
values. The cross-entropy is also widely used in classification.

## Kullback-Leiber divergence

The Kullback-Leiber divergence quantifies how a probability distribution P (for example in classification, the true 
datas) differs from the probability distribution Q. It measures the amount of information lost when Q is used to 
approximate P. Another way to see the divergence is how many more bits are needed for using a coding scheme optimized 
for the distribution Q. There are multiple applications for this divergence, notably the reinforcement learning in some
gradients methods like Trust Region Policy Optimization or Proximal Policy Optimization, to constrain the change in the 
policy between updates for keeping a stability in the learning process. Another use case is for Variational Autoencoders,
for regularizing the loss functions of VAEs.


