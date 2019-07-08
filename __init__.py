import torch

def init_cnn(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def init_rnn(m):
    if type(m) in [torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU]:
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                for ih in p.chunk(3, 0):
                    torch.nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in n:
                for hh in p.chunk(3, 0):
                    torch.nn.init.orthogonal_(hh)
            elif 'bias_ih' in n:
                torch.nn.init.ones_(p)
    elif type(m) in [torch.nn.LSTMCell, torch.nn.RNNCell, torch.nn.GRUCell]:
        for hh, ih in zip(m.weight_hh.chunk(3, 0), m.weight_ih.chunk(3, 0)):
            torch.nn.init.orthogonal_(hh)
            torch.nn.init.xavier_uniform_(ih)
        torch.nn.init.ones_(m.bias_ih)