## Part 1 : PerceptronAndGate

For the perceptron and gate, as we can't modify values in Haskell due to how pure functions works, we have
two solutions :

1) We can use monads to modify the values
2) We can use a recursive function that contains the history of weights and bias and add to the list the updated weights

I decided to use the recursive function because it's still a bit difficult for me to use monads, and because it's 
simpler when debugging because we can follow how the weights and bias are updated.

For my train function, I tried to translate the python version by using as much as possible the functions in hasktorch.
I had some problems when trying to manipulate the tensors dimensions because some of my tensors were TensorLike and 
they apparently don't implement dimensions.

Here are my results :

```text
Sum of initial errors: Tensor Float []  3.0000
Final weights: Tensor Float [1,2] [[ 2.0000   ,  4.0000   ]]
Final bias: Tensor Float [1,1] [[-4.1000   ]]
Final perceptronTensor Float [4,1] [[ 1.0000   ],
                    [ 0.0000],
                    [ 0.0000],
                    [ 0.0000]]
Final errorTensor Float []  0.0000
```
We can see that we go from 3 to 0 error value, so the code seems to work well after 60 epochs. 
There is also a problem with the calculation of my error, because I don't apply the absolute value the sum of the error 
can lead to a false 0 error at the end, which can stop the training before the end.


## Part 2 : MlpXOR.hs 

The first code uses MLPSpec for specifying the activation function and the number of neurons per layers. MLP contains
the linear layers themselves. The MLP function do the forward pass part by applying the weights and bias, then the activation 
function (here tanh). The main code generate random datas for the input of the xor gate, then apply mlp function to do the 
forward propagation, before calculating the loss and using gradiant descent to the model's state (backpropagation part).
It then updates weights and bias accordingly before going again until the model is optimized.

The second code uses sigmoid instead of the tanh function. In the second code, we don't generate random datas for the input,
and we calculate the loss in the entire trainingData instead, similarly to what I done for the loss calculation in my simple
perceptron. It also uses CUDA, that will probably lead to better performance as it can use optimized operations for Nvidia GPUs.

Using different activation function can lead to different results. For example, the step function won't work well because
the gradient descent uses partial differentiation, but the derivate of 0 and 1 is 0, so the weights won't update because of that, and
the learning will stop.

Others functions will work. The main difference between sigmoid and tanh is that tanh will converge faster due to being centered
in 0. These two functions however has the main problem of the vanishing gradient : because the gradient can go to smaller and smaller values, 
we can have the same problem than the step function, i.e gradient going to zero and stopping the learning.
