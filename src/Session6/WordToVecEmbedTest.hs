{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Session6.WordToVecEmbedTest where

import Torch.Tensor (Tensor, asTensor)
import Torch.Autograd (toDependent)
import Torch.NN (sample)
import Torch.Serialize (loadParams)
import Torch.Tensor (indexSelect)
import Torch.Functional(meanDim,cat)
import Torch.TensorFactories (randnIO')
import Torch.Autograd (makeIndependent, toDependent)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as B
import qualified Data.Map.Strict as M
import System.IO (hFlush, stdout)
import GHC.Generics (Generic)

import Torch.NN (Parameterized(..), Parameter,Randomizable(..),Linear,linear, LinearSpec(..))
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

-- Chemins
modelPath = "data/sample_embedding.params"
wordLstPath = "data/sample_wordlst.txt"


-- Fonction pour récupérer l'embedding
embeddingForWord :: Model -> [(String, Int)] -> String -> IO (Maybe Tensor)
embeddingForWord model word2idx word = do
  case lookup word word2idx of
    Just i -> do
      let embMatrix = toDependent $ wordEmbedding $ embeddings model
      let emb = indexSelect 0 (asTensor [i :: Int]) embMatrix
      return (Just emb)
    Nothing -> return Nothing
-- Main
main :: IO ()
main = do
  -- Chargement du vocabulaire (mot -> index)
  word2idx <- Bin.decodeFile wordLstPath :: IO [(String, Int)]
  let vocabSize = length word2idx
  let embDim = 9 -- ⚠️ Doit être identique à celui utilisé à l'entraînement

  -- Chargement du modèle
  let modelSpec = ModelSpec
        { embSpec = EmbeddingSpec vocabSize embDim
        , linSpec = LinearSpec embDim vocabSize
        }
  model <- sample modelSpec
  loadedModel <- loadParams model modelPath


  -- Entrée utilisateur
  putStr "Enter a word : "
  hFlush stdout
  word <- getLine

  -- Recherche de l'embedding
  maybeEmb <- embeddingForWord loadedModel word2idx word
  case maybeEmb of
    Just emb -> putStrLn $ "Embedding of the word \"" ++ word ++ "\":\n" ++ show emb
    Nothing  -> putStrLn $ "Can't find the word \"" ++ word ++ "\" in the vocabulary."
