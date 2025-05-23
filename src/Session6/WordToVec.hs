{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE OverloadedStrings #-}

module Session6.WordToVec (main) where
import Codec.Binary.UTF8.String (encode,decode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word (Word8)
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.Lazy.Encoding as TLE
import qualified Data.Text.Lazy as TL
import Data.List (nub,sort,sortOn)
import Data.Maybe (mapMaybe)
import Data.Char(toLower,isLetter, isAlphaNum, isSpace,isMark, generalCategory, GeneralCategory(NonSpacingMark,SpacingCombiningMark))

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding',nllLoss', logSoftmax,Dim(..),unsqueeze)
import Torch.NN (Parameterized(..), Parameter,Randomizable(..),Linear,linear, LinearSpec(..))
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor,shape,asValue,indexSelect)
import Torch.TensorFactories (eye', zeros',randnIO')
import Torch.Optim (Optimizer, runStep,Adam,mkAdam)
import System.Random (randomRIO)
import Control.Monad(forM_)
import Data.IORef
import ML.Exp.Chart (drawLearningCurve)

-- your text data (try small data first)
textFilePath = "data/sample_copy.txt"
modelPath =  "data/sample_embedding.params"
wordLstPath = "data/sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)


data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

-- Probably you should include model and Embedding in the same data class.
data Model = Model {
    embeddings :: Embedding,
    linear :: Linear
  } deriving (Show, Generic, Parameterized)

data ModelSpec = ModelSpec
  { embSpec :: EmbeddingSpec
  , linSpec :: LinearSpec
  } deriving (Show, Eq)


instance Randomizable EmbeddingSpec Embedding where
  sample EmbeddingSpec{..} = do
    emb <- makeIndependent =<< randnIO' [wordNum, wordDim]
    return $ Embedding emb

instance Randomizable ModelSpec Model where
  sample ModelSpec{..} = do
    emb <- sample embSpec
    let wordDim' = wordDim embSpec
        wordNum' = wordNum embSpec
    lin <- sample $ LinearSpec wordDim' wordNum'
    return $ Model emb lin
isUnncessaryChar ::
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!"]

shouldKeep :: Char -> Bool
shouldKeep c = isAlphaNum c || (isLetter c && not (isAscii c)) || isSpace c || c == '\''
  where
    -- Basic ASCII check
    isAscii ch = ch >= '\NUL' && ch <= '\DEL'


preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    filteredtexts = B.pack $ encode $  filter shouldKeep $ decode $ B.unpack texts
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0..]))

lowercase :: B.ByteString -> B.ByteString
lowercase bs = B.pack $ encode $ map toLower $ decode $ B.unpack bs


toyEmbedding ::
  EmbeddingSpec ->
  Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} =
  eye' wordNum wordDim

generateSkipGrams :: Int -> [B.ByteString] -> [(B.ByteString, B.ByteString)]
generateSkipGrams windowSize ws =
  concatMap (\(i, w) ->
    let contextIndices = [j | j <- [i - windowSize .. i + windowSize], j /= i, j >= 0, j < length ws]
    in map (\j -> (w, ws !! j)) contextIndices
  ) (zip [0..] ws)

safeDecode :: B.ByteString -> Maybe TL.Text
safeDecode bs = case TLE.decodeUtf8' bs of
    Right txt -> Just txt
    Left _    -> Nothing


vectorRepresentation :: [B.ByteString] -> [(B.ByteString, Int)]
vectorRepresentation listOfWords =
    let
        decodedPairs = mapMaybe (\bs -> fmap (\txt -> (bs, txt)) (safeDecode bs)) listOfWords
        sorted = sortOn (TL.toLower . snd) decodedPairs
        word2index = zip (map fst sorted) [0..]
    in word2index

lookupIndex :: B.ByteString -> [(B.ByteString, Int)] -> Maybe Int
lookupIndex token indexList = lookup token indexList

convertPairsToIndices
  :: [(B.ByteString, B.ByteString)]  -- (target, context)
  -> [(B.ByteString, Int)]           -- word2index
  -> ( [Int]  -- X_train: target indices
     , [Int]  -- y_train: context indices
     )
convertPairsToIndices pairs indexList =
  let indexedPairs = mapMaybe (\(t, c) -> do
                                tIdx <- lookupIndex t indexList
                                cIdx <- lookupIndex c indexList
                                return (tIdx, cIdx)
                              ) pairs
  in unzip indexedPairs



train :: Optimizer opt => Int -> Int -> Int -> Model -> opt -> Tensor -> Tensor -> IO [Float]
train numEpochs batchSize vocabSize model0 optimizer0 xTrainTensor yTrainTensor = do
  let numSamples = shape xTrainTensor !! 0

      loop 0 _model _opt accLosses = return (reverse accLosses)
      loop epoch model opt accLosses = do
        totalLossRef <- newIORef 0.0

        let loopBatch startIdx m o
              | startIdx >= numSamples = return (m, o)
              | otherwise = do
                  let endIdx = min (startIdx + batchSize) numSamples
                      batchIndices = asTensor [startIdx .. endIdx - 1] :: Tensor

                      xBatch = indexSelect 0 batchIndices xTrainTensor
                      yBatch = indexSelect 0 batchIndices yTrainTensor

                      embTensor = embedding' (toDependent $ wordEmbedding $ embeddings m) xBatch
                      linOut = Torch.NN.linear (Session6.WordToVec.linear m) embTensor
                      yPred = logSoftmax (Dim 1) linOut
                      loss = nllLoss' yBatch yPred

                  (newModel, newOpt) <- runStep m o loss 1e-3
                  let lossVal = asValue loss :: Float
                  modifyIORef totalLossRef (+ lossVal)

                  loopBatch (startIdx + batchSize) newModel newOpt

        (model', opt') <- loopBatch 0 model opt
        totalLoss <- readIORef totalLossRef
        if epoch `mod` 10 == 0
          then do
             putStrLn $ "Saving model to" ++ modelPath
             saveParams model' modelPath
             drawLearningCurve ("graph-loss-" ++ show epoch ++ "word2vec.png") "Training Loss Curve" [("loss", reverse(totalLoss:accLosses))]
          else return()
        putStrLn $ "Epoch: " ++ show (numEpochs - epoch + 1) ++ ", Total Loss: " ++ show totalLoss
        loop (epoch - 1) model' opt' (totalLoss : accLosses)

  loop numEpochs model0 optimizer0 []

main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath

  -- Create a unique word list
  let wordLines = preprocess texts
      allWords = concat wordLines
      wordFreqs = M.fromListWith (+) [(w,1:: Int)  | w <- allWords]
      topWords = take 30000 $ map fst $ sortOn (negate . snd) (M.toList wordFreqs)
      filteredWordLines = map (filter (`elem` topWords)) wordLines
      wordlst = filter (not . B.null) $ nub (concat filteredWordLines)

  let wordToIndex = wordToIndexFactory wordlst
      skipGrams = generateSkipGrams 2 wordlst
      vectors = vectorRepresentation wordlst
      (xTrainIdxs, yTrainIdxs) = convertPairsToIndices skipGrams vectors



  -- Tensor preparation
  let xTrainTensor = asTensor xTrainIdxs :: Tensor
      yTrainTensor = asTensor yTrainIdxs :: Tensor

  -- Define model spec
  let embDim = 9
      vocabSize = length wordlst
      modelSpec = ModelSpec
        { embSpec = EmbeddingSpec vocabSize embDim
        , linSpec = LinearSpec embDim vocabSize
        }
  print vocabSize

  model <- sample modelSpec
  let optimizer = Torch.Optim.mkAdam 0 0.9 0.999 (flattenParameters model)


  losses <- train 1000 128 vocabSize model optimizer xTrainTensor yTrainTensor
  drawLearningCurve "graph-loss-word2vec.png" "Training Loss Curve" [("loss", losses)]
  saveParams model modelPath
  return ()
