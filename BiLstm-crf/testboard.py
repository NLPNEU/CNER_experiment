import numpy as np
import pickle

if __name__ == '__main__':
    with open('/home/math007/work_sapce/fangcheng/ner/bilstm_crf/data/vocab.pkl', 'rb') as f:
        embed = pickle.load(f) 
    print(len(embed))

    with open('/home/math007/work_sapce/fangcheng/ner/bilstm_crf/data/embedding_matrix.pkl', 'rb') as f:
        data = pickle.load(f)
    print(len(data))
