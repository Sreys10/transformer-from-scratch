import torch
import torch.nn as nn
import math

##input embeddingss
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int ):
        super(). __init__()
        self.d_model= d_model
        self.vocab_size= vocab_size
        self.embedding= nn.Embedding(vocab_size. d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  #here we divide the embeddings with sqrt of dmodel as written in the paper
     


#positional Encodings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout= nn.Dropout(dropout)
        self.d_model= d_model
        self.seq_len= seq_len

        #create a matrix of shape(seq_len, d_model)
        pe= torch.zeros(seq_len, d_model)

        #create a vector od shape(seq_len)
        position= torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #(seq_len, 1)

        div_term= torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        #apply the sin to even pos
        pe[:, 0::2]= torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe=pe.unsqueeze(0) #(1, seq_len, d_model)


        self.register_buffer('pr', pe)

    def forward(self, x):
        x= x+ (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
    

#layer norm

class LayerNorm(nn.Module):
    def __init__(self, eps: float =10**-6) ->None:  #we use here eps for numerical stability
        super().__init__()
        self.eps= eps
        self.alpha= nn.Parameter(torch.ones(1)) #multipied
        self.bias= nn.Parameter(torch.zeros(1)) #added

    def forrward(self, x):
        mean= x.mean(dim=-1, keepdim=True)
        std= x.std(dim=-1, keepdim=True)

        x= self.alpha * (x- mean)/(std + self.eps) + self.bias #formulae
        return x
    
##feedforward network
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1= nn.Linear(d_model, d_ff) #first layer w1 nd b1
        self.linear2= nn.Linear(d_ff, d_model) #second layer w2 nd b2
        self.dropout= nn.Dropout(dropout)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    


##Multihead attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model= d_model   
        self.h= h
        assert d_model % h==0, "d_model is not divisible by h"

        self.d_k= d_model//h

        self.w_q=nn.Linear(d_model, d_model) #wq
        self.w_k= nn.Linear(d_model, d_model) #wk       
        self.w_v= nn.Linear(d_model, d_model) #wv

        self.w_o =nn.Linear(d_model, d_model) #wo
        self.dropout= nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  #scaled dot product attention
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  #softmax on last dim
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q, l, v, mask):
        query= self.w_q(q)         # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        key= self.w_k(k)
        value= self.w_v(v)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) --> (batch, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h,self.d_k).transpose(1,2)   
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)  

        #(Batch, h, seq_len, d_k) -->(Batch, Seq_Len, h, d_k) --> (Batch, Seq_len, d_model) 

        x= x.transpose(1,2).contiguose().view(x.shape[0], x.shape[1], -1, self.h* self.d_k) # (batch, seq_len, d_model)

        return self.w_o(x) 