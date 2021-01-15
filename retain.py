import numpy as np
import pickle
import random
import argparse
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim


def load_data(seqFile, labelFile):
    train_set_x = pickle.load(open(seqFile+'.train','rb'))
    valid_set_x = pickle.load(open(seqFile+'.valid','rb'))
    test_set_x = pickle.load(open(seqFile+'.test','rb'))
    
    train_set_y = pickle.load(open(labelFile+'.train','rb'))
    valid_set_y = pickle.load(open(labelFile+'.valid','rb'))
    test_set_y = pickle.load(open(labelFile+'.test','rb'))
    
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]), reverse=True)
    
    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]
    
    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]
    
    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]
    
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)
    
    return train_set, valid_set, test_set


def padMatrix(seqs, input_size):
    lengths = np.array([len(seq) for seq in seqs]).astype('int32')
    n_samples = len(seqs)
    
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, input_size))
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:,idx,:], seq):
            xvec[subseq] = 1.

    return x, lengths


def calculate_auc(test_model, dataset, batch_size, input_size):
    n_batches = int(np.ceil(float(len(dataset[0])) / float(batch_size)))
    scoreVec = []
    for index in range(n_batches):
        batchX = dataset[0][index*batch_size:(index+1)*batch_size]
        #pad matrix
        x, lengths = padMatrix(batchX, input_size)
        #get ouput from model
        x = torch.LongTensor(x)
        scores = test_model(x, lengths)
        scoreVec.extend(list(scores))
    labels = dataset[1]
    auc = roc_auc_score(list(labels), list(scoreVec))
    
    return auc


class RETAIN(nn.Module):
    def __init__(self, args):
        super(RETAIN, self).__init__()
        
        self.embed_size = args.emb_size
        self.emb = nn.Linear(args.n_input_codes, args.emb_size, bias=False)
        self.drop_embed = nn.Dropout(args.dropout_emb)
        
        self.alpha = nn.GRU(input_size=args.emb_size, hidden_size=args.alpha_dim)
        self.alpha_dense = nn.Linear(args.alpha_dim, 1)
        
        self.beta = nn.GRU(input_size=args.emb_size, hidden_size=args.beta_dim)
        self.beta_dense = nn.Linear(args.beta_dim, args.emb_size)
        
        self.output = nn.Linear(args.emb_size, 1)
        self.drop_context = nn.Dropout(args.dropout_context)
        
    def forward(self, x, lengths):
        
        x = x.float()
        x_emb = self.emb(x)
        x_emb = self.drop_embed(x_emb)
        
        #reverse visit sequence
        re_emb = torch.flip(x_emb, [0])
        
        alpha_out, alpha_h = self.alpha(re_emb)
        alpha_out = torch.flip(alpha_out, [0])
        alpha_out = nn.functional.softmax(self.alpha_dense(alpha_out))
        
        beta_out, beta_h = self.beta(re_emb)
        beta_out = torch.flip(beta_out, [0])
        beta_out = torch.tanh(self.beta_dense(beta_out))

        
        a1 = alpha_out.size(0)
        a2 = alpha_out.size(1)
        b1 = beta_out.size(0)
        b2 = beta_out.size(1)
        
        alpha = alpha_out.transpose(0,1).contiguous().view(-1,a1*a2)
        beta = beta_out.view(b1*b2,-1)
        
        c_t = alpha[:,:,None] * beta
        c_t = c_t.squeeze()
        c_t = (c_t * x_emb).sum(axis=0)
        
        c_vector = self.output(c_t)
        c_vector = self.drop_context(c_vector)
        preY = torch.sigmoid(c_vector)
        
        return preY
    
    def init_weight(self):
        
        self.emb.weight = torch.nn.init.uniform_(self.emb.weight, a=-0.1, b=0.1)
        self.alpha.weight_ih_l0 = torch.nn.init.uniform_(self.alpha.weight_ih_l0, a=-0.1, b=0.1)
        self.alpha.weight_hh_l0  = torch.nn.init.uniform_(self.alpha.weight_hh_l0, a=-0.1, b=0.1)
        self.alpha.bias_ih_l0 = torch.nn.init.zeros_(self.alpha.bias_ih_l0)
        self.alpha.bias_hh_l0 = torch.nn.init.zeros_(self.alpha.bias_hh_l0)
        self.beta.weight_ih_l0 = torch.nn.init.uniform_(self.beta.weight_ih_l0, a=-0.1, b=0.1)
        self.beta.weight_hh_l0  = torch.nn.init.uniform_(self.beta.weight_hh_l0, a=-0.1, b=0.1)
        self.beta.bias_ih_l0 = torch.nn.init.zeros_(self.beta.bias_ih_l0)
        self.beta.bias_hh_l0 = torch.nn.init.zeros_(self.beta.bias_hh_l0)
        
        self.alpha_dense.weight = torch.nn.init.uniform_(self.alpha_dense.weight, a=-0.1, b=0.1)
        self.alpha_dense.bias = torch.nn.init.zeros_(self.alpha_dense.bias)
        self.beta_dense.weight = torch.nn.init.uniform_(self.beta_dense.weight, a=-0.1, b=0.1)
        self.beta_dense.bias = torch.nn.init.zeros_(self.beta_dense.bias)
        
        self.output.weight = torch.nn.init.uniform_(self.output.weight, a=-0.1, b=0.1)
        self.output.bias = torch.nn.init.zeros_(self.output.bias)
        
        #for consistent weights
        '''
        left = -0.1
        right = 0.1
        
        np.random.seed(1234)
        self.emb.weight = torch.nn.Parameter(torch.from_numpy(np.random.uniform(left, right, (128, 942))))
        self.alpha.weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.random.uniform(left, right, (3*128, 128))))
        self.alpha.weight_hh_l0  = torch.nn.Parameter(torch.from_numpy(np.random.uniform(left, right, (3*128, 128))))
        self.alpha.bias_ih_l0 = torch.nn.init.zeros_(self.alpha.bias_ih_l0)
        self.alpha.bias_hh_l0 = torch.nn.init.zeros_(self.alpha.bias_hh_l0)
        self.beta.weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.random.uniform(left, right, (3*128, 128))))
        self.beta.weight_hh_l0  = torch.nn.Parameter(torch.from_numpy(np.random.uniform(left, right, (3*128, 128))))
        self.beta.bias_ih_l0 = torch.nn.init.zeros_(self.beta.bias_ih_l0)
        self.beta.bias_hh_l0 = torch.nn.init.zeros_(self.beta.bias_hh_l0)
        
        self.alpha_dense.weight = torch.nn.Parameter(torch.from_numpy(np.random.uniform(left, right, (1, 128))))
        self.alpha_dense.bias = torch.nn.init.zeros_(self.alpha_dense.bias)
        self.beta_dense.weight = torch.nn.Parameter(torch.from_numpy(np.random.uniform(left, right, (128,128))))
        self.beta_dense.bias = torch.nn.init.zeros_(self.beta_dense.bias)
        
        self.output.weight = torch.nn.Parameter(torch.from_numpy(np.random.uniform(left, right, (1, 128))))
        self.output.bias = torch.nn.init.zeros_(self.output.bias)
        '''
        
       
def train_RETAIN(model, trainset, validset, testset, args):
    
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.1, 0.001), weight_decay=args.L2, eps=1e-8)
    
    n_batches = int(np.ceil(float(len(trainset[0]))/float(args.batch_size)))
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch= 0
    
    print("Optimization Start")
    for epoch in range(args.epochs):
        model.train()
        
        trial = 1
        for index in random.sample(range(n_batches), n_batches):
            
            weights = []
            for param in model.parameters():
                weights.append(param.clone())
                
            batchX = trainset[0][index*args.batch_size:(index+1)*args.batch_size]
            y = trainset[1][index*args.batch_size:(index+1)*args.batch_size]
            batchX, lengths = padMatrix(batchX, args.n_input_codes)
            batchX = torch.LongTensor(batchX)
            y = torch.LongTensor(y)
            
            optimizer.zero_grad()
            out = model(batchX, lengths)
            out = out.squeeze()
            
            y = y.type_as(out)
            cost = loss(out, y)
            cost.backward()
            optimizer.step()
            
            
            #check the updates of weights
            '''
            weights_after_backprop = []
            for param in model.parameters():
                weights_after_backprop.append(param.clone())
                
            param_name = []
            for name, param in model.named_parameters():
                param_name.append(name)
                
            
            print("trial: ", trial)
            for i in zip(param_name, weights, weights_after_backprop):
                print(i[0], ": ",torch.equal(i[1], i[2]))
            '''
            
            trial += 1
            
        else:
            with torch.no_grad():
                model.eval()
                validAuc = calculate_auc(model, validset, args.batch_size, args.n_input_codes)
                print("Epoch: %d, Validation Aucc: %f" %(epoch, validAuc))

                
                if (validAuc > bestValidAuc):
                    bestValidAuc = validAuc
                    bestValidEpoch = epoch
                    bestTestAuc = calculate_auc(model, testset, args.batch_size, args.n_input_codes)
                    print('Current Best Validation Aucc Found. Test Aucc: %f at epoch: %d' %(bestTestAuc, bestValidEpoch))
                    torch.save(model, args.out_file + '/model.'+str(epoch)+',pth')
                    
    print('Best Validation & Test Aucc: %f, %f at epoch %d' %(bestValidAuc, bestTestAuc, bestValidEpoch))
                    

def parse_arguments(parser):
    parser.add_argument('--seq_file', type=str, required=True, help='path to the pickled file containing patient visit information') #seqeunce file
    parser.add_argument('n_input_codes', type=int, required=True, help='numver of unique input medical codes') #n_input_codes
    parser.add_argument('label_file', type=str, required=True, help='path to the pickled file containg patient label information') #label file
    parser.add_argument('out_file', type=str, required=True, help='path of directory the output models will be saved') #output file
    parser.add_argument('--emb_size', type=int, default=128, help='dimension size of embedding layer')
    parser.add_argument('--alpha_dim', type=int, default=128, help='alpha hidden layer dimension size') #alpha hidden dimension size
    parser.add_argument('--beta_dim', type=int, default=128, help='beta hidden layer dimension size') #beta hidden dimension size
    parser.add_argument('--batch_size', type=int, default=100, help='batch size') #batch size
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs') #epochs
    parser.add_argument('--L2', type=float, default=0.0001, help='L2 regularization value') #L2
    parser.add_argument('--dropout_emb', type=float, default=0.0, help='dropout rate for embedding') #embedding dropout
    parser.add_argument('--dropout_context', type=float, default=0.0, help='dropout rate for contect vector') #context dropout
    
    args = parser.parse_args
    return args


def main(args):
    print("Loading Data ... ")
    trainset, validset, testset = load_data(args.seq_file, args.label_file)
    print("Done")
    
    print("Building Model ... ")
    model = RETAIN(args)
    model.init_weight()
    
    print("Training Model ... ")
    train_RETAIN(model, trainset, validset, testset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    
    main(args)