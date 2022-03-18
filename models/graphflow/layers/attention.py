'''
Created on Nov, 2018

@author: hugo

'''
import torch
import torch.nn as nn

import warnings
# warnings.simplefilter("error")
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", Warning)

INF = 1e20
class Context2QuestionAttention(nn.Module):
    def __init__(self, dim, hidden_size):
        super(Context2QuestionAttention, self).__init__()
        self.linear_sim = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, context, questions, out_questions, ques_mask=None):
        """
        Parameters
        :context, (batch_size, ?, ctx_size, dim)
        :questions, (batch_size, turn_size, ques_size, dim)
        :out_questions, (batch_size, turn_size, ques_size, ?)
        :ques_mask, (batch_size, turn_size, ques_size)

        Returns
        :ques_emb, (batch_size, turn_size, ctx_size, dim)
        """
        # shape: (batch_size, ?, ctx_size, dim), ? equals 1 or turn_size
        context_fc = torch.relu(self.linear_sim(context))
        # shape: (batch_size, turn_size, ques_size, dim)
        questions_fc = torch.relu(self.linear_sim(questions))

        # shape: (batch_size, turn_size, ctx_size, ques_size)
        attention = torch.matmul(context_fc, questions_fc.transpose(-1, -2))
        if ques_mask is not None:
            # print("Context2Question Attention")
            # print(1 - ques_mask.byte().unsqueeze(2))
            # print((1 - ques_mask.byte().unsqueeze(2)).to(torch.bool))
            # print((1 - ques_mask.byte().unsqueeze(2)).to(torch.bool).dtype)
            mask = (1 - ques_mask.byte().unsqueeze(2)).to(torch.bool)
            attention = attention.masked_fill_(mask, -INF)
        prob = torch.softmax(attention, dim=-1)
        # shape: (batch_size, turn_size, ctx_size, ?)
        ques_emb = torch.matmul(prob, out_questions)
        return ques_emb

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W1 = torch.Tensor(input_size, hidden_size)
        self.W1 = nn.Parameter(nn.init.xavier_uniform_(self.W1))
        self.W2 = torch.Tensor(hidden_size, 1)
        self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))

    def forward(self, x, attention_mask=None):
        attention = torch.mm(torch.tanh(torch.mm(x.view(-1, x.size(-1)), self.W1)), self.W2).view(x.size(0), -1)
        if attention_mask is not None:
            # Exclude masked elements from the softmax
            # print("Self Attention")
            # print(1-attention_mask.byte())
            # print((1-attention_mask.byte()).to(torch.bool).dtype)
            # print(type((1-attention_mask.byte()).to(torch.bool)))
            mask = (1-attention_mask.byte()).to(torch.bool)
            attention = attention.masked_fill_(mask, -INF)

        probs = torch.softmax(attention, dim=-1).unsqueeze(1)
        weighted_x = torch.bmm(probs, x).squeeze(1)
        return weighted_x
