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
We can see that we go from 3 to 0 error value, so the code seems to work. However, I noticed after some executions that
I don't have everytime the correct values for the perceptron : sometimes I still have errors at the end.
There is also a problem with the calculation of my error, because I don't apply the absolute value the sum of the error 
can lead to a false 0 error at the end, which can stop the training before the end.
