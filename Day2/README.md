# Informations on homework for the session 3

We first convert the numbers to tensor by using asTensor.


FIRST EXECUTION : 
Then, we can zip the two list, and looping through them. We could use foldl, but here we don't need the accumulator,
and we also want to print the result, so we use IO actions. Because I chosed to do everything in the mapping function,
we don't need to return anything, so we choose mapM_ for that.

So I had :
```haskell

  mapM_
    ( \(x,y) -> do
        let estimatedY = linear (sampleA, sampleB) (asTensor [x])
        putStrLn("correct = " ++ show y)
        putStrLn("estimated = " ++ show estimatedY)
        putStrLn("********")
    )
    (zip xsList ysList)
```

Then I realized that I could apply linear to the entire list if I use mul instead of matmul (we can't use matmul as 
slope and inputs has a different shape). That way, we can just get the list of estimatedY instead, which will be useful
for the loss calculation.

For the loss calculation, we use Mean Squared Error, as we want to exacerbate the cost. 
We can use the functions in Torch.Functional for calculating the cost.

For gradient descent, I didn't manage to have the correct gradient with the numeric gradient, so I tried the analytic method, by applying the partial derivate of the cost function with intercept and slope. I don't think my implementation is correct, because I didn't manage to find the correct learningRate/epoch values 


--- Final Predictions vs Actual ---
Correct = 130.0
Estimated = 153.16309
********
Correct = 195.0
Estimated = 192.47119
********
Correct = 218.0
Estimated = 288.6726
********
Correct = 166.0
Estimated = 185.23022
********
Correct = 163.0
Estimated = 223.5039
********
Correct = 155.0
Estimated = 131.44019
********
Correct = 204.0
Estimated = 157.30078
********
Correct = 270.0
Estimated = 202.81543
********
Correct = 205.0
Estimated = 130.40576
********
Correct = 127.0
Estimated = 80.75342
********
Correct = 260.0
Estimated = 218.33179
********
Correct = 249.0
Estimated = 267.98413
********
Correct = 251.0
Estimated = 263.84644
********
Correct = 158.0
Estimated = 119.0271
********
Correct = 167.0
Estimated = 179.02368

Got that after final training :
Epoch 995468: Slope (a)=0.5592, Intercept (b)=93.8070, Cost=1117.4471 (corrected it because in my first training I had the good values as initial values.... )

I guess there is a problem with one of my functions, because I need to do around 1 million epoch before approaching the good values. But if I put a learning rate more than 28e-6, I diverge to + infinity. 