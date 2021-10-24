import csv,sys
import queue
import matplotlib.pyplot as plt

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

        if idx % 32 != 0:
            continue

        negedge_q.put(v_value)
        if negedge_q.qsize() > 10:
            negedge_q.get()

            # extract negedge edge
            try :
                if negedge_q.queue[0] - negedge_q.queue[-1] >= 0.75 and prev_can_signal_len != len(can_signal) :
                    NEGEDGE = True
                    #print(len(can_signal), negedge_q.queue[0], negedge_q.queue[-1])
                    prev_can_signal_len = len(can_signal)
                if NEGEDGE == True:
                    negedge_term += 1
                if negedge_term >= 5:
                    for q_item in negedge_q.queue:
                        negedge_list.append(q_item)
                        #print("Negedge Edge: ", q_item, len(can_signal))
                    NEGEDGE = False
                    negedge_term = 0
                    negedge_q.empty()
                    #print("=================================")
            except IndexError :
                continue

# visualize
plt.plot(negedge_list)
plt.ylim(0, 4)
plt.ylabel('Voltage [V]')
plt.xlabel('Sample')
plt.show()
