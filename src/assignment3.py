# -*- coding: utf-8 -*-
"""
    Deep Learning for NLP
    Assignment 3: Language Identification using Recurrent Architectures
    Based on original code by Hande Celikkanat & Miikka Silfverberg
"""


from random import choice, random, shuffle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk

from data import read_datasets, WORD_BOUNDARY, UNK, HISTORY_SIZE
from paths import data_dir

torch.set_num_threads(10)



#--- hyperparameters ---
N_EPOCHS = 100
LEARNING_RATE = 0.0001
REPORT_EVERY = 5
EMBEDDING_DIM = 30
HIDDEN_DIM = 20
BATCH_SIZE = 64
N_LAYERS = 1


#--- models ---
class LSTMModel(nn.Module):
    '''
    model should have:
    1 embbeding layer
    1 one recurrent layer
    1 one feed-forward layer
    + log softmax non linearity

    '''
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(LSTMModel, self).__init__()

        self.embedding_dim = embedding_dim              # 30
        self.character_set_size = character_set_size    # 73
        self.n_layers = n_layers        # layer dimensions - just 1
        self.hidden_dim = hidden_dim    # hidden dimensions 20
        self.n_classes = n_classes      # 6 possibilities

        # embedding [73 X 30]
        #self.embedding = nn.Embedding(self.character_set_size, self.embedding_dim, max_norm=True)
        # embedding [73 X 30]
        self.embedding = nn.Embedding(self.character_set_size, self.embedding_dim)

        # building the LSTM  [30 X 1 X 20]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers)

        # output layer (hidden_dim X n_classes) [20 X 6]
        self.fc = nn.Linear(hidden_dim, n_classes)


    def forward(self, inputs):
        # WRITE CODE HERE
        # Embedding input shape: [len_char X 1]

        embeds = self.embedding(inputs)
        # 'Embedding output shape: [len_char X batch_size X 30]

        # Recommendation: use a single input for lstm layer
        # (no special initialization of the hidden layer):
        lstm_out, (hidden) = self.lstm(embeds, None)
        # LSTM output shape: [len_char X batch_size X 20]

        # get the final state of the hidden layer
        lstm_out = lstm_out[-1]
        # shape of final state of the recurrent layer:  [batch_size X 20]

        # output pass
        output = self.fc(lstm_out)
        # Output shape [batch_size X 6]

        #  get the class log probabilities
        output = F.log_softmax(output, dim=1)
        # Final Output shape [batch_size X 6]
        return output


class GRUModel(nn.Module):
    '''
    A Gated Recurrent Unit (GRU), as its name suggests, is a variant of the RNN architecture,
     and uses gating mechanisms to control and manage the flow of information between cells
     in the NN.
    '''
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(GRUModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # embedding [73 X 30]
        self.embedding = nn.Embedding(self.character_set_size, self.embedding_dim, max_norm=True)

        # building the GRU [30 X 20 X 1]
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers)

        # output layer (hidden_dim X n_classes) [20 X 6]
        self.fc = nn.Linear(hidden_dim, n_classes)


    def forward(self, inputs):
        # WRITE CODE HERE
        embeds = self.embedding(inputs) # output shape: [len_char X batch_size X 30]

        # Recommendation: use a single input for gru layer
        # (no special initialization of the hidden layer):
        gru_out, hidden = self.gru(embeds, None) # shape: [len_char X batch_size X 20]

        # get the final state of the recurrent layer
        output = gru_out[-1]
        # shape of final state of the recurrent layer:  [batch_size X 20]

        # output pass
        output = self.fc(output)
        # Output shape [batch_size X 6]

        #  get the class log probabilities
        output = F.log_softmax(output, dim=1)
        # Final Output shape [batch_size X 6]
        return output


class RNNModel(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(RNNModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # embedding [73 X 30]
        self.embedding = nn.Embedding(self.character_set_size, self.embedding_dim, max_norm=True)

        # building the GRU [30 X 1 X 20]
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers)

        # output layer (hidden_dim X n_classes) [20 X 6]
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, inputs):
        # WRITE CODE HERE
        # initialize hidden
        hidden_0 = torch.randn(self.n_layers, inputs.size(1), self.hidden_dim)  # hidden state

        embeds = self.embedding(inputs)  # output shape: [len_char X batch_size X 30]

        # Recommendation: use a single input for rnn layer (no special initialization of the hidden layer):
        rnn_out, hidden = self.rnn(embeds, hidden_0)

        # shape of final state of the recurrent layer:  [batch_size X 20]
        # output pass
        output = self.fc(rnn_out[-1])
        # Output shape [batch_size X 6]

        #  get the class log probabilities
        output = F.log_softmax(output, dim=1)
        # Final Output shape [batch_size X 6]
        # WRITE MORE CODE HERE
        return output



# --- auxilary functions ---
def get_minibatch(minibatchwords, character_map, languages):
    # INPUTS
    mb_x_padding_lenght = len(minibatchwords[-1]['TENSOR'])
    batch_size = len(minibatchwords)
    mb_x = torch.empty((mb_x_padding_lenght, batch_size ), dtype=torch.long)
    # LABEL
    mb_y = torch.empty(batch_size, dtype=torch.long)

    for idx, dictionary in enumerate(minibatchwords):
        mb_x_i = dictionary['TENSOR']          # get characters tensor
        len_mb_x_i = len(dictionary['TENSOR']) # get len tensor
        mb_x[:, idx] = F.pad(mb_x_i, (0, mb_x_padding_lenght - len_mb_x_i), 'constant', character_map['#'])

        mb_y_i = dictionary['LANGUAGE'] # get label
        mb_y[idx] = label_to_idx(mb_y_i, languages)

    return mb_x, mb_y


def label_to_idx(lan, languages):
    languages_ordered = list(languages)
    languages_ordered.sort()
    return torch.LongTensor([languages_ordered.index(lan)])


def get_word_length(word_ex):
    return len(word_ex['WORD'])


def evaluate(dataset, model, eval_batch_size, character_map, languages):
    correct = 0
    validation_loss = 0
    model.eval()
    
    # WRITE CODE HERE IF YOU LIKE
    for i in range(0,len(dataset),eval_batch_size):
        minibatchwords = dataset[i:i+eval_batch_size]    
        mb_x, mb_y = get_minibatch(minibatchwords, character_map, languages)

        # like training but without optimization
        outputs = model(mb_x)

        # make prediction
        y_preds = []
        for out in outputs:
            y_preds.append((torch.argmax(out)).item())

        targets = []
        for y in mb_y:
            targets.append(y.item())

        # compare predictions to target
        correct_batch=0
        for ind, v in enumerate(y_preds):
            if v == targets[ind]:
                correct_batch += 1

        correct += correct_batch

    return correct * 100.0 / len(dataset)



if __name__=='__main__':

    # --- select the recurrent layer according to user input ---
    if len(sys.argv) < 2:
        print('-------')
        print('You didn''t provide any arguments!')
        print('Using LSTM model as default')
        print('To select a model, call the program with one of the arguments: -lstm, -gru, -rnn')
        print('Example: python assignment3_LOAICIGA.py -gru')
        print('-------')
        model_choice = 'lstm'
    elif len(sys.argv) == 2:
        print('-------')
        print('Running with ' + sys.argv[1][1:] + ' model')
        print('-------')        
        model_choice = sys.argv[1][1:]
    else:
        print('-------')
        print('Wrong number of arguments')
        print('Please call the model with exactly one argument, which can be: -lstm, -gru, -rnn')
        print('Example: python assignment3_LOAICIGA.py -gru')
        print('Using LSTM model as default')
        print('-------')        
        model_choice = 'lstm'


    #--- initialization ---

    if BATCH_SIZE == 1:
        data, character_map, languages = read_datasets('uralic.mini',data_dir)
    else:
        data, character_map, languages = read_datasets('uralic',data_dir)

    trainset = [datapoint for lan in languages for datapoint in data['training'][lan]]
    n_languages = len(languages)
    character_set_size = len(character_map)

    model = None
    if model_choice == 'lstm':
        model = LSTMModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)
    elif model_choice == 'gru':
        model = GRUModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)
    elif model_choice == 'rnn':
        model = RNNModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.NLLLoss()


    # --- training loop ---
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0

        # Generally speaking, it's a good idea to shuffle your
        # datasets once every epoch.
        shuffle(trainset)

        # WRITE CODE HERE
        # Sort your training set according to word-length, 
        # so that similar-length words end up near each other
        # You can use the function get_word_length as your sort key.
        trainset.sort(key=lambda item: len(item.get('WORD')))


        for i in range(0,len(trainset),BATCH_SIZE):
            minibatchwords = trainset[i:i+BATCH_SIZE]

            #print(minibatchwords)

            mb_x, mb_y = get_minibatch(minibatchwords, character_map, languages)

            # forward pass
            outputs = model(mb_x)

            # Clear gradients
            optimizer.zero_grad()

            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_function(outputs, mb_y)
            total_loss += loss.item()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
            
        print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))

        if ((epoch+1) % REPORT_EVERY) == 0:
            train_acc = evaluate(trainset,model,BATCH_SIZE,character_map,languages)
            dev_acc = evaluate(data['dev'],model,BATCH_SIZE,character_map,languages)
            print('epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' % 
                  (epoch+1, total_loss, train_acc, dev_acc))

        
    # --- test ---    
    test_acc = evaluate(data['test'],model,BATCH_SIZE,character_map,languages)        
    print('test acc: %.2f%%' % (test_acc))
