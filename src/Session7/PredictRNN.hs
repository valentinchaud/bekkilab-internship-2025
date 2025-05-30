{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}


module Session7.PredictRNN (main) where
import Codec.Binary.UTF8.String (encode,decode) -- add utf8-string to dependencies in package.yaml
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import Data.List(tails)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics
import Torch.NN (Parameter, Parameterized(..), forward, Randomizable(..),Linear(..),linear,LinearSpec(..))
import Torch.Serialize (loadParams)
import Torch.TensorFactories (randnIO',zeros')
import Torch.Autograd (makeIndependent)
import Torch.Layer.RNN
import Torch.Device (Device(..), DeviceType(..))
import Torch.Tensor (sliceDim,size,Tensor,select,reshape,indexSelect,asTensor,asValue,shape)
import Torch.Layer.NonLinear(ActName(..))
import qualified Data.Map.Strict as M
import Torch.Optim (Optimizer, runStep,Adam,mkAdam)
import Torch.Functional(nllLoss',transpose,Dim(..))
import Data.IORef
import Data.Char(toLower,isLetter, isAlphaNum, isSpace,isMark, generalCategory, GeneralCategory(NonSpacingMark,SpacingCombiningMark))
import ML.Exp.Chart (drawLearningCurve)
-- amazon review data
data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic)

instance FromJSON Image
instance ToJSON Image

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
  } deriving (Show, Generic)

instance FromJSON AmazonReview
instance ToJSON AmazonReview

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)


data ModelSpec = ModelSpec {
  embSpec :: EmbeddingSpec,
  linSpec :: LinearSpec,
  rnnSpec :: RnnHypParams
} deriving (Show, Eq, Generic)


data Model = Model {
  emb :: Embedding,
  rnn :: RnnParams,
  linear:: Linear
} deriving (Show, Generic, Parameterized)
instance Randomizable EmbeddingSpec Embedding where
  sample EmbeddingSpec{..} = do
    emb <- makeIndependent =<< randnIO' [wordNum, wordDim]
    return $ Embedding emb


instance Randomizable ModelSpec Model where
  sample ModelSpec{..} = do
    emb <- sample embSpec
    rnn <- sample rnnSpec
    lin <- sample linSpec
    return $ Model emb rnn lin
initialize ::
  ModelSpec ->
  FilePath ->
  IO Model
initialize modelSpec embPath = do
  -- Initialiser le modèle avec des poids aléatoires
  randomizedModel <- sample modelSpec

  return randomizedModel

amazonReviewPath :: FilePath
amazonReviewPath = "data/train.jsonl"

outputPath :: FilePath
outputPath = "data/review-texts.txt"

embeddingPath =  "data/sample_embedding.params"

wordLstPath = "data/sample_wordlst.txt"

decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview]
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
  in sequenceA $ map eitherDecode jsonList
forward :: Model -> Tensor -> Tensor -> Tensor
forward Model{..} h0 inputs =
  let
    seqLen = Torch.Tensor.size 0 inputs
    inputs' = Torch.Functional.transpose (Dim 0) (Dim 1) inputs  -- (seq_len, batch_size)
    (rnnOut, _) = rnnLayers rnn Tanh Nothing h0 inputs'
    lastHidden = select 0 (seqLen - 1) rnnOut
    reshaped = reshape [-1] lastHidden
  in
    Torch.NN.forward linear reshaped

preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    filteredtexts = B.pack $ encode $  filter shouldKeep $ decode $ B.unpack texts
    textLines = B.split (head $ encode "\n") filteredtexts
shouldKeep :: Char -> Bool
shouldKeep c = isAlphaNum c || (isLetter c && not (isAscii c)) || isSpace c || c == '\''
  where
    -- Basic ASCII check
    isAscii ch = ch >= '\NUL' && ch <= '\DEL'


wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0..]))

lowercase :: B.ByteString -> B.ByteString
lowercase bs = B.pack $ encode $ map toLower $ decode $ B.unpack bs
train :: Optimizer opt => Int -> Int -> Int -> Int -> Model -> opt -> Tensor -> Tensor -> IO [Float]
train numEpochs batchSize hiddenSize vocabSize model0 optimizer0 xTrainTensor yTrainTensor = do
  let numSamples = shape xTrainTensor !! 0
      -- Utiliser les valeurs fixes basées sur la configuration dans main
      nLayers = 1  -- Selon la définition dans main
      isBidi = False  -- Selon la définition dans main

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

                      batchSz = min batchSize (endIdx - startIdx)
                      dirFactor = if isBidi then 2 else 1
                      h0 = zeros' [nLayers * dirFactor, batchSz, hiddenSize]

                      yPred = Session7.PredictRNN.forward model h0 xBatch
                      loss = nllLoss' yBatch yPred

                  (newModel, newOpt) <- runStep m o loss 1e-3
                  let lossVal = asValue loss :: Float
                  modifyIORef totalLossRef (+ lossVal)

                  loopBatch (startIdx + batchSize) newModel newOpt

        (model', opt') <- loopBatch 0 model opt
        totalLoss <- readIORef totalLossRef
        putStrLn $ "Epoch: " ++ show (numEpochs - epoch + 1) ++ ", Total Loss: " ++ show totalLoss
        loop (epoch - 1) model' opt' (totalLoss : accLosses)

  loop numEpochs model0 optimizer0 []
main :: IO ()
main = do
  jsonl <- B.readFile amazonReviewPath
  let amazonReviews = decodeToAmazonReview jsonl
      reviews = case amazonReviews of
                  Left err -> []
                  Right reviews -> reviews
      limitedReviews = take 1000 reviews
  -- Extraire le texte des reviews
  let reviewTexts = map text limitedReviews
      combinedText = B.pack $ encode $ unlines reviewTexts

  -- Prétraiter les textes
  let wordLines = preprocess combinedText
      allWords = concat wordLines

  -- Charger la liste de mots
  wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
  let wordCount = length wordLst
      wordToIndex = wordToIndexFactory wordLst

  -- Préparer les séquences pour le RNN en vérifiant que chaque liste est assez longue
  let seqLength = 10
      prepareSequences line =
        let lists = tails line
            validLists = filter (\xs -> length xs > seqLength) lists
        in [(take seqLength xs, xs !! seqLength) | xs <- validLists]

      wordLabelPairs = [ (line, rating rev) | (line, rev) <- zip wordLines limitedReviews, length line > seqLength ]
      sequences = [ (take seqLength line, score) | (line, score) <- wordLabelPairs ]
      -- Convertir en indices
  let (xSeq, ySeq) = unzip sequences
      xIndices = map (map wordToIndex) xSeq
      yTrainTensor = asTensor ySeq :: Tensor  -- shape: [batch_size], e.g., Float labels

  -- Convertir en tenseurs
  let xTrainTensor = asTensor xIndices :: Tensor
      yTrainTensor = asTensor ySeq :: Tensor

  -- Configuration du modèle
  let embSpec = EmbeddingSpec {
        wordNum = wordCount,
        wordDim = 9
      }
  let rnnSpec = RnnHypParams {
    dev = Device CPU 0,
    bidirectional = False,
    inputSize = 9,
    hiddenSize = 50,
    numLayers = 1,
    hasBias = True
  }

  let modelSpec = ModelSpec {
      embSpec = embSpec,
      linSpec = LinearSpec 9 wordCount,
      rnnSpec = rnnSpec
    }

  -- Initialiser le modèle et l'optimiseur
  initModel <- initialize modelSpec embeddingPath
  let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initModel)

  -- Entraîner le modèle
  putStrLn "Début de l'entraînement du RNN..."
  losses <- train 5 9 50 wordCount initModel optimizer xTrainTensor yTrainTensor

  -- Afficher la courbe d'apprentissage
  drawLearningCurve "graph-loss-rnn.png" "Courbe d'entraînement RNN" [("loss", losses)]

  putStrLn "Entraînement terminé !"
