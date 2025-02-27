# Neural Language Modeling - Next Word Prediction

## Introduction
This project implements three neural language models for Next Word Prediction (NWP) using PyTorch:
- **Feed Forward Neural Network (FFNN)**
- **Vanilla Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**

Each model is trained on two corpora:  
1. *Pride and Prejudice* (124,970 words)  
2. *Ulysses* (268,117 words)  

The models predict the most probable next word given an input sentence.

---


## File Structure

```
├── generator.py         # Main script for next word prediction
├── models/              # (not uploaded - please access using link in the report)
│   ├── ffnn_model.pt    # Pretrained FFNN model
│   ├── rnn_model.pt     # Pretrained RNN model
│   ├── lstm_model.pt    # Pretrained LSTM model 
├── data/external
│   ├── pride_prejudice.txt  # Pride and Prejudice corpus
│   ├── ulysses.txt          # Ulysses corpus for data processing
├── README.md            # Instructions for running the project
├── report.pdf           # Analysis of results
```

## How to Run

### 1. Generating Next Word Predictions

Run the `generator.py` script with the following command:

```sh
python3 generator.py <lm_type> <corpus_path> <k>
```

Where:
- `<lm_type>`: Choose between `-f` (FFNN), `-r` (RNN), or `-l` (LSTM)
- `<corpus_path>`: Path to the dataset file (e.g., `./corpus/PrideAndPrejudice.txt`)
- `<k>`: Number of top word predictions to display

#### Example Usage:

```sh
python3 generator.py -f ./corpus/PrideAndPrejudice.txt 3
```

#### Example Output:

```
Input sentence: An apple a day keeps the doctor
Output:
away 0.4
happy 0.2
fresh 0.1
```

### Pretrained Models

The following pretrained models are provided in the models folder:

- **FFNN Model**
- **RNN Model**
- **LSTM Model**

To load a pretrained model in PyTorch:

```python
import torch
model = torch.load("models/<corpus name>_<lm_type>_model_<n_gram>_gram.pt")
model.eval()
```
where, <corpus_name> = pride or ulysses, <lm_type> = ffnn or rnn or lstm
<br>
## Implementation Assumptions

- The FFNN model uses n-grams (n=3 and n=5) for training.
- The RNN and LSTM models process variable-length input sequences.
- A test set of 1,000 randomly selected sentences is used for evaluation.
- Perplexity is computed for training and test sets to compare model performance.

## Evaluation & Analysis

A detailed report (`report.pdf`) includes:

- Perplexity scores for each model on both training and test sets.
- Model ranking based on performance.
- Observations on:
  - Performance on longer sentences.
  - The effect of n-gram size on FFNN.
  - The comparison between neural and statistical models.

## References

- [Neural Networks and Deep Learning]([Link](http://neuralnetworksanddeeplearning.com/index.html))
- [How to Create a Neural Network](https://www.youtube.com/watch?v=hfMk-kjRv4c)
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Stanford NLP Book - RNNs and LSTMs](https://web.stanford.edu/~jurafsky/slp3/9.pdf)
```

You can access the link to the pretrained models given at the end of the report.
