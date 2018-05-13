import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import random
import time
import math

MIN_LENGTH = 3
MAX_LENGTH = 16 # determines output vector dimension
MAX_NUMBER = 2**8 # determines input vector dimension
INPUT_TENSOR_LENGTH = math.ceil(math.log2(MAX_NUMBER))
OUTPUT_TENSOR_LENGTH = MAX_LENGTH + 1
teacher_forcing_ratio = 0.5 # higher means more teacher forcing

DATA_SIZE = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = [0 for i in range(OUTPUT_TENSOR_LENGTH)]
EOS_token = 0 # Position 0 means end of sequence

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__() 
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):  
        output = input.view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
        
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
 
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):  
        output = F.relu(input.view(1, 1, -1)) 
        output, hidden = self.gru(output, hidden) 
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
        
        


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def sort(arr):
    """
    return an array of the sorted positions of the original array. The positions
    are sorted based on the value of the integer on that position.
    value to large value
    
    Note that position starts from 1, not 0
    Example:
    arr:                   2 3 1 4
    Sorted Array:          1 2 3 4
    Sorted Positions :     3 1 2 4
    after an array is sorted
    """
    pos_arr = []
    dict = {}
    for i in range(len(arr)):
        x = arr[i]
        if x in dict:
            dict[x].append(i)
        else:
            dict[x] = [i] 
    for x in sorted(arr):
        pos_arr.append(dict[x][0] + 1)
        del dict[x][0] 
    
    return pos_arr
    

def prepareData(number):
    data_pairs = [] 
    for i in range(number):
        size = np.random.randint(MIN_LENGTH,MAX_LENGTH)
        arr = np.random.rand(size) * MAX_NUMBER
        arr = list(np.array(arr, dtype=int))
        data_pairs.append((arr, sort(arr)))
    
    return data_pairs


 
def intToBin(x): 
    bin_string = bin(x)[2:]
    binary = [0] * INPUT_TENSOR_LENGTH
    for i in range(1, 1+len(bin_string)):
        binary[-i] = int(bin_string[-i])
    return binary
    
def intToOneHot(x):
    """
    x represents a position (output vector)
    """
    onehot = torch.zeros(OUTPUT_TENSOR_LENGTH)
    onehot[x] = 1
    return onehot

def tensorsFromPair(pair): 
    input_list = [intToBin(x) for x in pair[0]]   
    output_list = pair[1]
    return (torch.FloatTensor(input_list, device = device), torch.LongTensor(output_list, device = device))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden() 
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.FloatTensor([SOS_token], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
 
            loss += criterion(decoder_output, target_tensor[di:di+1])
            decoder_input = intToOneHot(target_tensor[di].item())  
            # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di:di+1])
            if decoder_input.item() == EOS_token:
                break
            decoder_input = intToOneHot(decoder_input.item())

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

    
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    data_pairs = prepareData(DATA_SIZE) 
    print(random.choice(data_pairs))
    training_pairs = [tensorsFromPair(random.choice(data_pairs))
                      for i in range(n_iters)]
    print("Finish Preparing all training data")

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#
 


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, pair, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor, target_tensor = tensorsFromPair(pair)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.FloatTensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_sequence = [] 

        for di in range(len(pair[0])):
            decoder_output, decoder_hidden= decoder(
                decoder_input, decoder_hidden) 
            topv, topi = decoder_output.data.topk(1)
            #print(topi.item())
            if topi.item() == EOS_token or topi.item() > len(pair[0]):
                return decoded_sequence
            else:
                decoded_sequence.append(pair[0][topi.item()-1])
 
            decoder_input = intToOneHot(topi.item())
        return decoded_sequence


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=10):
    correct = 0
    correct_per_length = {i:0 for i in range(MIN_LENGTH, MAX_LENGTH+1)}
    count_per_length = {i:0 for i in range(MIN_LENGTH, MAX_LENGTH+1)}
    data_pairs = prepareData(n) 
    for i in range(n):
        pair = data_pairs[i]
        sorted_arr = sorted(pair[0])
        print('>', pair[0])
        print('=', sorted_arr)
        output_array = evaluate(encoder, decoder, pair) 
        print('<', output_array)
        print('')
        if output_array == sorted(pair[0]):
            correct += 1
            correct_per_length[len(pair[0])] += 1
        count_per_length[len(pair[0])] += 1
        print("Total Success Rate %.2f%%"%(correct/(i+1) * 100))
    for size, correct in correct_per_length.items():
        total = count_per_length[size]
        if total != 0:
            print("Success Rate for N = %d is %.2f%%"%(size, correct/total * 100))

######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

hidden_size = 256
encoder1 = EncoderRNN(INPUT_TENSOR_LENGTH, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, OUTPUT_TENSOR_LENGTH).to(device)
learning_rate_multiplier = [1, 1, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1, 0.05, 0.05, 0.025,0.025, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.0025, 0.0025, 0.001, 0.001, 0.0005, 0.0005, 0.00025, 0.00025, 0.0001, 0.0001]
for epoch in range(26, 27):
    trainIters(encoder1, decoder1, 75000, print_every=100, learning_rate = 0.01 *learning_rate_multiplier[epoch])

######################################################################
#
torch.save(encoder1, "model/e4")
torch.save(decoder1, "model/d4")
evaluateRandomly(encoder1, decoder1, 1000)