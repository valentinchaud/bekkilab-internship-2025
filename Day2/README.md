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
We can use the functions in Torch.Functional for calculating the cost
