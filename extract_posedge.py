import csv,sys
import queue
from statistics import mean, median, variance, stdev
from scipy.stats import skew, kurtosis
from functools import reduce
from math import sqrt
import numpy as np

def rms(xs):
    return sqrt(reduce(lambda a, x: a + x * x, xs, 0) / len(xs))

def en(xs):
    return reduce(lambda a, x: a + x * x, xs, 0) / len(xs)

idx = 0
posedge_term = 0
threshold = 2.97
can_dom_bit_idx = 0 # 0 to 2000
can_res_bit_idx = 0 # 0 to 2000
can_signal = []
SOF = False
POSEDGE = False
posedge_q = queue.Queue()
posedge_list = []
prev_can_signal_len = -1

with open(sys.argv[1]) as f:
    while True:
        idx += 1
        row = f.readline()
        if idx == 1 or idx == 2 or idx == 3: 
            continue
        #print(row, end='')
        try :
            v_value = float(row.split(',')[1])
        except IndexError:
            break

        # estimate can bit signal
        if v_value >= threshold :
            #print(v_value)
            SOF = True
            can_dom_bit_idx += 1
        elif SOF == True and v_value < threshold :
            can_res_bit_idx += 1
        if can_dom_bit_idx == 1000:
            can_signal.append('0')
        elif can_res_bit_idx == 1000:
            can_signal.append('1')
        elif can_res_bit_idx >= 2000: 
            can_res_bit_idx = 0
        elif can_dom_bit_idx >= 2000: 
            can_dom_bit_idx = 0

        posedge_q.put(v_value)
        if posedge_q.qsize() > 200:
            posedge_q.get()

            # extract posedge edge
            try :
                if posedge_q.queue[-1] - posedge_q.queue[0] >= 0.5 and prev_can_signal_len != len(can_signal) :
                    POSEDGE = True
                    #print(len(can_signal), posedge_q.queue[0], posedge_q.queue[-1])
                    prev_can_signal_len = len(can_signal)
                if POSEDGE == True:
                    posedge_term += 1
                if posedge_term >= 50:
                    for q_item in posedge_q.queue:
                        posedge_list.append(q_item)
                        #print("Posedge Edge: ", q_item, len(can_signal))
                    POSEDGE = False
                    posedge_term = 0
                    posedge_q.empty()
                    #print("=================================")
            except IndexError :
                continue

# feature extraction
fft_posedge_list = abs(np.fft.fft(posedge_list))
print('{:.4f}'.format(mean(posedge_list)),\
      '{:.4f}'.format(stdev(posedge_list)),\
      '{:.4f}'.format(variance(posedge_list)),\
      '{:.4f}'.format(skew(posedge_list)),\
      '{:.4f}'.format(kurtosis(posedge_list)),\
      '{:.4f}'.format(max(posedge_list)),\
      '{:.4f}'.format(min(posedge_list)),\
      '{:.4f}'.format(rms(posedge_list)),\
      '{:.4f}'.format(en(posedge_list)),\
      '{:.4f}'.format(mean(fft_posedge_list)),\
      '{:.4f}'.format(stdev(fft_posedge_list)),\
      '{:.4f}'.format(variance(fft_posedge_list)),\
      '{:.4f}'.format(skew(fft_posedge_list)),\
      '{:.4f}'.format(kurtosis(fft_posedge_list)),\
      '{:.4f}'.format(max(fft_posedge_list)),\
      '{:.4f}'.format(min(fft_posedge_list)),\
      '{:.4f}'.format(rms(fft_posedge_list)),\
      '{:.4f}'.format(en(fft_posedge_list))
)
