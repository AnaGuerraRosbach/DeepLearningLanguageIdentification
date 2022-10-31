import os
from collections import defaultdict as dd
import nltk
import torch
import io

WORD_BOUNDARY='#'
UNK='UNK'
HISTORY_SIZE=4

def read_words(fn):
    data = []
    for line in io.open(fn, encoding="utf8"):
        line = line.strip('\n')
        if line:
            wf, lan = line.split('\t')
            data.append({'WORD':wf, 
                         'TOKENIZED WORD':[c for c in wf],
                         'LANGUAGE':lan})
    return data 

def compute_tensor(word_ex,charmap):
    word_ex['TENSOR'] = torch.LongTensor([charmap[WORD_BOUNDARY]]
                                     + [charmap[c] if c in charmap 
                                                   else charmap[UNK] 
                                                   for c in word_ex['TOKENIZED WORD']]
                                     + [charmap[WORD_BOUNDARY]])

def sort_by_language(dataset):
    sorteddata = dd(lambda : [])
    for ex in dataset:
        sorteddata[ex['LANGUAGE']].append(ex)
    return sorteddata

def read_datasets(prefix,data_dir):
    datasets = {'training': read_words(os.path.join(data_dir, '%s.%s' % 
                                                    (prefix, 'train'))), 
                'dev': read_words(os.path.join(data_dir, '%s.%s' % 
                                               (prefix, 'dev'))),
                'test': read_words(os.path.join(data_dir, '%s.%s' %
                                                (prefix, 'test')))} 

    charmap = {c:i for i,c in enumerate({c for ex in datasets['training'] 
                                         for c in ex['TOKENIZED WORD']})}
    charmap[UNK] = len(charmap)
    charmap[WORD_BOUNDARY] = len(charmap)
    languages = {ex['LANGUAGE'] for ex in datasets['training']}

    for word_ex in datasets['training']:
        compute_tensor(word_ex,charmap)
    for word_ex in datasets['dev']:
        compute_tensor(word_ex,charmap)
    for word_ex in datasets['test']:
        compute_tensor(word_ex,charmap)

    datasets['training'] = sort_by_language(datasets['training'])

    return datasets, charmap, languages

if __name__=='__main__':
    from paths import data_dir
    d,c = read_datasets('uralic',data_dir)


