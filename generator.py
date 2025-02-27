import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from tokenizer import Tokenizer
import sys
import os

# Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_gram):
        super(FFNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(n_gram * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.n_gram = n_gram

    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_gram):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.n_gram = n_gram

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_output = output[:, -1, :]
        logits = self.fc(last_output)
        return logits

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_gram):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.n_gram = n_gram

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        last_output = output[:, -1, :]
        logits = self.fc(last_output)
        return logits

class NGramDataset(Dataset):
    def __init__(self, ngrams, vocab):
        self.ngrams = ngrams
        self.vocab = vocab

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        context, target = self.ngrams[idx]
        context_tensor = torch.tensor([self.vocab.get(word, 0) for word in context], dtype=torch.long)
        target_tensor = torch.tensor(self.vocab.get(target, 0), dtype=torch.long)
        return context_tensor, target_tensor

def generate_ngrams(tokenized_text, n):
    ngrams = []
    for sentence in tokenized_text:
        if len(sentence) < n:
            continue
        for i in range(len(sentence) - n + 1):
            context = sentence[i:i + n-1]
            target = sentence[i + n-1]
            ngrams.append((context, target))
    return ngrams

def train_model(model, dataloader, vocab, inv_vocab, epochs=20, lr=0.001, model_path=None, k=5):
    # model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            # context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
        
        with torch.no_grad():
            sample_context, sample_target = next(iter(dataloader))
            # sample_context, sample_target = sample_context.to(device), sample_target.to(device) 
            sample_output = model(sample_context)
            probs = torch.softmax(sample_output, dim=1)
            top_k = torch.topk(probs, k, dim=1)
            
            print(f"Sample Input: {[inv_vocab[idx.item()] for idx in sample_context[0].cpu()]}")
            print(f"Actual Target: {inv_vocab[sample_target[0].item()]}")
            print("Top-k Predictions:")
            for i in range(k):
                word = inv_vocab[top_k.indices[0][i].item()]
                prob = top_k.values[0][i].item()
                print(f"{word}: {prob:.4f}")
            print("-" * 50)
    
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'inv_vocab': inv_vocab
        }, model_path)
        print(f"Model and vocab saved to {model_path}")
        
def compute_perplexity(model, sentences, vocab, n_gram, batch_size=256):
    # model.to(device)  
    model.eval()
    criterion = nn.CrossEntropyLoss()
    perplexities = []
    processed_sentences = []
    total_skipped = 0

    with torch.no_grad():
        for sentence in sentences:
            if len(sentence) < n_gram:
                total_skipped += 1
                continue
                
            original_sentence = ' '.join(sentence)
            processed_sentences.append(original_sentence)
            
            sentence_loss = 0
            num_targets = 0

            for i in range(0, len(sentence) - n_gram+1, batch_size):
                batch_contexts = []
                batch_targets = []
                for j in range(i, min(i + batch_size, len(sentence) - n_gram+1)):
                    context = sentence[j:j + n_gram-1]
                    target = sentence[j + n_gram-1]
                    context_tensor = torch.tensor([vocab.get(word, 0) for word in context], dtype=torch.long)
                    target_tensor = torch.tensor(vocab.get(target, 0), dtype=torch.long)
                    batch_contexts.append(context_tensor)
                    batch_targets.append(target_tensor)
                if not batch_contexts:
                    continue
                batch_contexts = torch.stack(batch_contexts)
                batch_targets = torch.stack(batch_targets)
                output = model(batch_contexts)
                loss = criterion(output, batch_targets)
                sentence_loss += loss.item() * len(batch_contexts)
                num_targets += len(batch_contexts)
            if num_targets == 0:
                continue

            avg_loss = sentence_loss / num_targets
            perplexity = torch.exp(torch.tensor(avg_loss))
            perplexities.append(perplexity.item())

    if total_skipped > 0:
        print(f"Warning: Skipped {total_skipped} sentences shorter than {n_gram} tokens.")

    if not perplexities:
        return float('inf'), []
    
    average_perplexity = sum(perplexities) / len(perplexities)
    return average_perplexity, perplexities, processed_sentences

def save_perplexity_reports(corpus_name, lm_type, n_gram, split_type, sentences, perplexities, average_perplexity):
    base_dir = os.path.join("perplexity_scores", split_type)
    os.makedirs(base_dir, exist_ok=True)
    
    filename = f"{corpus_name}_{lm_type}_{n_gram}_gram.txt"
    report_path = os.path.join(base_dir, filename)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Perplexity Report ({split_type.capitalize()})\n")
        f.write(f"Corpus: {corpus_name} | Model: {lm_type} | N-gram: {n_gram}\n")
        f.write("=" * 50 + "\n")
        
        f.write(f"\nAverage Perplexity: {average_perplexity:.4f}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("Individual Sentence Scores:\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (sentence, perplexity) in enumerate(zip(sentences, perplexities)):
            f.write(f"Sentence {i+1}:\n")
            f.write(f"Text: {sentence}\n")
            f.write(f"Perplexity: {perplexity:.4f}\n\n")
    
    print(f"Saved {split_type} perplexity scores to: {report_path}")


def main(pretrained_model=False):
    
    if len(sys.argv) < 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        sys.exit(1)
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])
    n_gram = 3
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = Tokenizer()
    tokenized_text = tokenizer.tokenize(text)
    
    random.seed(42)
    random.shuffle(tokenized_text)
    test_sentences = tokenized_text[:1000]
    train_sentences = tokenized_text[1000:]
    
    corpus_filename = os.path.basename(corpus_path).lower()

    if corpus_filename.startswith("pride"):
        model_path = None
        if lm_type == '-f':
            model_path = f"models/pride_ffnn_model_{n_gram}_gram.pt"
        elif lm_type == '-r':
            model_path = f"models/pride_rnn_model_{n_gram}_gram.pt"
        elif lm_type == '-l':
            model_path = f"models/pride_lstm_model_{n_gram}_gram.pt"
        else:
            print("Invalid model type")
            sys.exit(1)
    elif corpus_filename.startswith("ulysses"):
        model_path = None
        if lm_type == '-f':
            model_path = f"models/ulysses_ffnn_model_{n_gram}_gram.pt"
        elif lm_type == '-r':
            model_path = f"models/ulysses_rnn_model_{n_gram}_gram.pt"
        elif lm_type == '-l':
            model_path = f"models/ulysses_lstm_model_{n_gram}_gram.pt"
        else:
            print("Invalid model type")
            sys.exit(1)
    
    vocab = None
    inv_vocab = None
    model = None
    
    if pretrained_model:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            vocab = checkpoint['vocab']
            inv_vocab = checkpoint['inv_vocab']
            if lm_type == '-f':
                model = FFNNLanguageModel(len(vocab), 100, 200, n_gram-1)
            elif lm_type == '-r':
                model = RNNLanguageModel(len(vocab), 100, 200, n_gram-1) 
            elif lm_type == '-l':
                model = LSTMLanguageModel(len(vocab), 100, 200, n_gram-1)
            model.load_state_dict(checkpoint['model_state_dict'])
            # model.to(device)
            print(f"Loaded pretrained model from {model_path}")
        else:
            print(f"No pretrained model found at {model_path}. Training from scratch.")
            pretrained_model = False
    
    if not pretrained_model:
        words = sorted(set(sum(tokenized_text, [])))
        vocab = {word: idx for idx, word in enumerate(words)}
        inv_vocab = {idx: word for word, idx in vocab.items()}
    
        if lm_type == '-f':
            model = FFNNLanguageModel(len(vocab), 100, 200, n_gram-1)
        elif lm_type == '-r':
            model = RNNLanguageModel(len(vocab), 100, 200, n_gram-1)
        elif lm_type == '-l':
            model = LSTMLanguageModel(len(vocab), 100, 200, n_gram-1)
        else:
            print("Invalid model type")
            sys.exit(1)
        # model.to(device)
    
    train_ngrams = generate_ngrams(train_sentences, n_gram)
    test_ngrams = generate_ngrams(test_sentences, n_gram)
    
    train_dataset = NGramDataset(train_ngrams, vocab)
    test_dataset = NGramDataset(test_ngrams, vocab)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    if not pretrained_model:
        train_model(model, train_loader, vocab, inv_vocab, epochs=10, lr=0.001, model_path=model_path, k=k)
    train_sentences_filtered = [s for s in train_sentences if len(s) >= n_gram]
    test_sentences_filtered = [s for s in test_sentences if len(s) >= n_gram]

    # corpus_name = os.path.basename(corpus_path).split('.')[0].lower()
    
    # # For training data
    # train_avg_perplexity, train_perplexities, train_sentences_processed = compute_perplexity(
    #     model, train_sentences_filtered, vocab, n_gram
    # )
    # save_perplexity_reports(corpus_name, lm_type, n_gram, "train", 
    #                        train_sentences_processed, train_perplexities, train_avg_perplexity)
    
    # # For test data
    # test_avg_perplexity, test_perplexities, test_sentences_processed = compute_perplexity(
    #     model, test_sentences_filtered, vocab, n_gram
    # )
    # save_perplexity_reports(corpus_name, lm_type, n_gram, "test", 
    #                        test_sentences_processed, test_perplexities, test_avg_perplexity)


    while True:
        sentence = input("Input sentence: ").strip().split()
        required_context_length = n_gram - 1
        if len(sentence) < required_context_length:
            print(f"Sentence is too short. Needs at least {required_context_length} words.")
            continue
        context = sentence[-required_context_length:]
        context_tensor = torch.tensor([vocab.get(word, 0) for word in context], dtype=torch.long).unsqueeze(0)
        output = model(context_tensor)
        probs = torch.softmax(output, dim=1).squeeze()
        top_k = torch.topk(probs, k)
        for idx, prob in zip(top_k.indices, top_k.values):
            print(f"{inv_vocab[idx.item()]} {prob.item():.4f}")

if __name__ == "__main__":
    main(pretrained_model=True)