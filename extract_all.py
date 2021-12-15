import csv,sys
import queue
from statistics import mean, median, variance, stdev
from scipy.stats import skew, kurtosis
from functools import reduce
from math import sqrt,floor,ceil
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
dominant_list = []
prev_can_signal_len = -1
sampling_rate = int(sys.argv[2])
skip_duration = floor(float(1/sampling_rate)*1000/2)
queue_length = ceil(sampling_rate/2.0)+1
buffering_term = floor((sampling_rate/2.0+1)/2)
dominant_buffering_term = 0 if sampling_rate==1 else ceil((sampling_rate/2.0+1)*1.5)

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
            can_dom_bit_idx += 2
        elif SOF == True and v_value < threshold :
            can_res_bit_idx += 2
        if can_dom_bit_idx == 1000:
            can_signal.append('0')
        elif can_res_bit_idx == 1000:
            can_signal.append('1')
        elif can_res_bit_idx >= 2000: 
            can_res_bit_idx = 0
        elif can_dom_bit_idx >= 2000: 
            can_dom_bit_idx = 0

        if idx % skip_duration != 0:
            continue

        posedge_q.put(v_value)
        if posedge_q.qsize() > queue_length:
            posedge_q.get()

            # extract posedge edge
            try :
                if posedge_q.queue[-1] - posedge_q.queue[0] >= 0.5 and prev_can_signal_len != len(can_signal) :
                    POSEDGE = True
                    #print(len(can_signal), posedge_q.queue[0], posedge_q.queue[-1])
                    prev_can_signal_len = len(can_signal)
                if POSEDGE == True:
                    posedge_term += 1
                if posedge_term >= dominant_buffering_term and POSEDGE == True:
                    if sampling_rate==1:
                        dominant_list.append(posedge_q.queue[-1])
                        #print("Dominant signals: ", q_item, len(can_signal))
                    else:
                        for q_item in posedge_q.queue:
                            dominant_list.append(q_item)
                            #print("Dominant signals: ", q_item, len(can_signal))
                    POSEDGE = False
                    posedge_term = 0
                    posedge_q.empty()
                    #print("=================================")
            except IndexError :
                continue

# label
#print('mean,stdev,variance,skew,kurtosis,max,min,rms,en,mean_fft,stdev_fft,variance_fft,skew_fft,kurtosis_fft,max_fft,min_fft,rms_fft,en_fft')
# feature extraction
fft_dominant_list = abs(np.fft.fft(dominant_list))

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
            can_dom_bit_idx += 2
        elif SOF == True and v_value < threshold :
            can_res_bit_idx += 2
        if can_dom_bit_idx == 1000:
            can_signal.append('0')
        elif can_res_bit_idx == 1000:
            can_signal.append('1')
        elif can_res_bit_idx >= 2000: 
            can_res_bit_idx = 0
        elif can_dom_bit_idx >= 2000: 
            can_dom_bit_idx = 0

        if idx % skip_duration != 0:
            continue

        posedge_q.put(v_value)
        if posedge_q.qsize() > queue_length:
            posedge_q.get()

            # extract posedge edge
            try :
                if posedge_q.queue[-1] - posedge_q.queue[0] >= 0.5 and prev_can_signal_len != len(can_signal) :
                    POSEDGE = True
                    #print(len(can_signal), posedge_q.queue[0], posedge_q.queue[-1])
                    prev_can_signal_len = len(can_signal)
                if POSEDGE == True:
                    posedge_term += 1
                if posedge_term >= buffering_term and POSEDGE == True:
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

idx = 0
negedge_term = 0
threshold = 2.97
can_dom_bit_idx = 0 # 0 to 2000
can_res_bit_idx = 0 # 0 to 2000
can_signal = []
SOF = False
NEGEDGE = False
negedge_q = queue.Queue()
negedge_list = []
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
            can_dom_bit_idx += 2
        elif SOF == True and v_value < threshold :
            can_res_bit_idx += 2
        if can_dom_bit_idx == 1000:
            can_signal.append('0')
        elif can_res_bit_idx == 1000:
            can_signal.append('1')
        elif can_res_bit_idx >= 2000: 
            can_res_bit_idx = 0
        elif can_dom_bit_idx >= 2000: 
            can_dom_bit_idx = 0

        if idx % skip_duration != 0:
            continue

        negedge_q.put(v_value)
        if negedge_q.qsize() > queue_length:
            negedge_q.get()

            # extract negedge edge
            try :
                if negedge_q.queue[0] - negedge_q.queue[-1] >= 0.5 and prev_can_signal_len != len(can_signal) :
                    NEGEDGE = True
                    #print(len(can_signal), negedge_q.queue[0], negedge_q.queue[-1])
                    prev_can_signal_len = len(can_signal)
                if NEGEDGE == True:
                    negedge_term += 1
                if negedge_term >= buffering_term and NEGEDGE == True:
                    for q_item in negedge_q.queue:
                        negedge_list.append(q_item)
                        #print("Negedge Edge: ", q_item, len(can_signal))
                    NEGEDGE = False
                    negedge_term = 0
                    negedge_q.empty()
                    #print("=================================")
            except IndexError :
                continue

# feature extraction
fft_negedge_list = abs(np.fft.fft(negedge_list))

print('{:.4f}'.format(mean(dominant_list)),\
      '{:.4f}'.format(stdev(dominant_list)),\
      '{:.4f}'.format(variance(dominant_list)),\
      '{:.4f}'.format(skew(dominant_list)),\
      '{:.4f}'.format(kurtosis(dominant_list)),\
      '{:.4f}'.format(max(dominant_list)),\
      '{:.4f}'.format(min(dominant_list)),\
      '{:.4f}'.format(rms(dominant_list)),\
      '{:.4f}'.format(en(dominant_list)),\
      '{:.4f}'.format(mean(fft_dominant_list)),\
      '{:.4f}'.format(stdev(fft_dominant_list)),\
      '{:.4f}'.format(variance(fft_dominant_list)),\
      '{:.4f}'.format(skew(fft_dominant_list)),\
      '{:.4f}'.format(kurtosis(fft_dominant_list)),\
      '{:.4f}'.format(max(fft_dominant_list)),\
      '{:.4f}'.format(min(fft_dominant_list)),\
      '{:.4f}'.format(rms(fft_dominant_list)),\
      '{:.4f}'.format(en(fft_dominant_list)),\
      
      '{:.4f}'.format(mean(negedge_list)),\
      '{:.4f}'.format(stdev(negedge_list)),\
      '{:.4f}'.format(variance(negedge_list)),\
      '{:.4f}'.format(skew(negedge_list)),\
      '{:.4f}'.format(kurtosis(negedge_list)),\
      '{:.4f}'.format(max(negedge_list)),\
      '{:.4f}'.format(min(negedge_list)),\
      '{:.4f}'.format(rms(negedge_list)),\
      '{:.4f}'.format(en(negedge_list)),\
      '{:.4f}'.format(mean(fft_negedge_list)),\
      '{:.4f}'.format(stdev(fft_negedge_list)),\
      '{:.4f}'.format(variance(fft_negedge_list)),\
      '{:.4f}'.format(skew(fft_negedge_list)),\
      '{:.4f}'.format(kurtosis(fft_negedge_list)),\
      '{:.4f}'.format(max(fft_negedge_list)),\
      '{:.4f}'.format(min(fft_negedge_list)),\
      '{:.4f}'.format(rms(fft_negedge_list)),\
      '{:.4f}'.format(en(fft_negedge_list)),\

      '{:.4f}'.format(mean(posedge_list)),\
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
      '{:.4f}'.format(en(fft_posedge_list)),\
      sys.argv[3],sep=','
)
