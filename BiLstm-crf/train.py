import torch
import torch.nn as nn
import torch.optim as optim
from model import *
import pickle
from tqdm import tqdm


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 300
HIDDEN_DIM = 50


with open('/home/math007/work_sapce/fangcheng/ner/bilstm_crf/data/train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('/home/math007/work_sapce/fangcheng/ner/bilstm_crf/data/dev.pkl', 'rb') as f:
    dev_data   = pickle.load(f)

def get_entity_pricision(pred_y, true_y):
    #TODO
    #give every label of the word, output the precision of the entity
    pass

def get_entity_recall(pred_y, true_y):
    #TODO
    #give every label of the word, output the recall of the entity
    pass

def get_entity_f1(pred_y, true_y):
    #TODO
    #get the f1 score of entity
    pass

with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
print('vocab size: ', len(vocab))

word_to_idx = {} #TODO put the embedding matrix here
for i, word in enumerate(vocab):
    word_to_idx[word] = i
print(len(word_to_idx))

tag_to_idx = {'O': 0, 'B-LOC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-ORG': 5, 'I-PER': 6, START_TAG: 7, STOP_TAG: 8}

model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay = 1e-4)

with open('./data/train.pkl', 'rb') as f:
    training_data = pickle.load(f)


if __name__ == '__main__':
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_idx)
        precheck_tags = torch.tensor([tag_to_idx[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    for epoch in range(5):
        for sentence, tags in tqdm(training_data):
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_idx)
            targets = torch.tensor([tag_to_idx[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_idx)
        print(model(precheck_sent))

