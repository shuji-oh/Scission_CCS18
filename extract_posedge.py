import csv,sys
import queue

idx = 0
posedge_term = 0
threshold_top = 3.0
threshold = 2.97
threshold_buttom = 2.7
can_dom_bit_idx = 0 # 0 to 2000
can_res_bit_idx = 0 # 0 to 2000
can_signal = []
SOF = False
POSEDGE = False
posedge_q = queue.Queue()

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

        try :
            # extract positive edge
            if threshold_top >= v_value >= threshold_buttom and can_signal[-1] == '1':
                #print(v_value)
                POSEDGE = True
            if POSEDGE == True:
                print(v_value)
                posedge_q.put(v_value)
                posedge_term += 1
            if posedge_term >= 100:
                POSEDGE = False
                posedge_term = 0
                print(posedge_q)
                posedge_q.empty()
        except IndexError :
            continue