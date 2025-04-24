module Session3.LinearRegression where
import Torch.Tensor (Tensor,asValue, asTensor,numel,shape)
import Torch.Functional (matmul,mul, add, transpose2D,sumAll,pow,sub,mean)
import Torch.Functional.Internal (meanAll)
ys :: Tensor
ys = asTensor([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])
xs :: Tensor
xs = asTensor([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

linear ::
        (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
        Tensor ->           -- ^ data x: 1 × 10
        Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = add intercept (mul slope input)

cost ::
    Tensor -> -- ^ ground truth: 1 × 10
    Tensor -> -- ^ estimated values: 1 × 10
    Tensor    -- ^ loss: scalar
cost z z' =  mean $ pow (2 :: Float)  (sub z z')

calculateNewA :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
calculateNewA x y slope intercept =
    let predicatedValues = linear (slope, intercept) x
        diff = sub predicatedValues y
    in mean (2 * diff * x)

calculateNewB :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
calculateNewB x y slope intercept =
    let predicatedValues = linear (slope, intercept) x
        diff = sub predicatedValues y
    in mean (2 * diff * x)