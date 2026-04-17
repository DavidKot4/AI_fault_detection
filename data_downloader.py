
import csv
import time
import modbus_polling
from pynput import keyboard
from pymodbus.client import ModbusTcpClient

#--CONFIG--
NAME="test.csv" #FAULTYPE_L1S_L2S_L3S_LEN 
STEP_SIZE=0.050
DEVICE_IP="192.168.168.29"
SAMPLE_LENGTH=20 #Total amount to save
HEADERS = ['t+','class','V_L1', 'V_L2', 'V_L3', 'V_L12', 'V_L23', 'V_L31', 'A_L1', 'A_L2', 'A_L3', 'VA_L1', 'VA_L2', 'VA_L3', 'W_L1', 'W_L2', 'W_L3', 'Q_L1', 'Q_L2', 'Q_L3', 'PF_L1', 'PF_L2', 'PF_L3', 'THD_L1', 'THD_L2', 'THD_L3']
FAULT_MAP= {
    '1': 1, #1-phase
    '2': 2, #2-phase
    '3': 3, #2-phase-ground
    '4': 4, #3-phase 
}

#--KEYBOARD TRACKING--
current_label = 0

def on_press(key):
    global current_label
    try:
        current_label = FAULT_MAP.get(key.char, 0)
        print(f"Key Recorded: {key.char}")
    except AttributeError:
        current_label = 0 #reset label to normal

def on_release(key):
    global current_label
    current_label = 0

#--DATA SAVING--
#createCSV
def create_csv(name, headers):
    with open(name, "x", newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"Created CSV file: {name}")

#save row to CSV
def save_row(file, data, curr_time, current_label):
    
    with open(file, 'a', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow([f"{curr_time:.3f}", current_label, *data,])

    print(f"Added row: t+{curr_time} to file")

#--EXECUTION--

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

client = ModbusTcpClient(DEVICE_IP, port=502)
client.connect()

create_csv(NAME, HEADERS)

start_time = time.time()
elapsed = 0
index = 0

print(f"Recording {SAMPLE_LENGTH}s of data.")

try:
    while elapsed < SAMPLE_LENGTH:
        loop_start = time.time()

        curr_row = modbus_polling.poll_device(client, DEVICE_IP)

        save_row(NAME, curr_row, elapsed, current_label)
        index += 1
        elapsed = index * STEP_SIZE

        #precise 20ms of sleep if faster
        time_to_sleep = STEP_SIZE - (time.time() - loop_start)
        if time_to_sleep > 0:   
            time.sleep(time_to_sleep)

except KeyboardInterrupt:
    print("Stop triggered by user")

print("Finished Collection")
client.close()
listener.stop()