import pickle
from tqdm import tqdm

def pre_process(input_path, output_path):
    train_data = []
#     punctuation = ['，', '。', '？', '！']
    with open(input_path, 'r') as f:
        data = f.readlines()
    tmp_word  = []
    tmp_label = []
    for i, word in enumerate(data):
        if not word.isspace(): 
            tmp_word.append(word.split()[0])
            tmp_label.append(word.split()[1])
        elif word.isspace():
            train_data.append((tmp_word, tmp_label))
            tmp_word = []
            tmp_label = []
            
    with open(output_path, 'wb') as f:
        pickle.dump(train_data, f)

if __name__ == '__main__':
    pre_process('/home/math007/data/chinese_ner/example.train', './data/train.pkl')
    pre_process('/home/math007/data/chinese_ner/example.dev', './data/dev.pkl')
    pre_process('/home/math007/data/chinese_ner/example.test', './data/test.pkl')

    with open('/home/math007/data/chinese_embedding/char_embedding') as f:
        embed = f.readlines() 

    vocab = ['OOV']
    embedding_matrix = [[0] * 300]
    for data in tqdm(embed[1:]):
       if len(data.split()) == 301 and (data.split()[0] not in vocab) and (len(data.split()[0]) == 1):
           try:
               embedding_matrix.append([float(x) for x in data.split()[1:]])
               vocab.append(data.split()[0])
           except:
               print(data)
    
    print("vocab size: ", len(vocab))
    print("embedding_matrix size: ", len(vocab))

    with open('./data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    with open('./data/embedding_matrix.pkl', 'wb') as f:
        pickle.dump(embedding_matrix, f)
