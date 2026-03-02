#output: Read in Register data as numpy array
#goal - process data into AI-trainable data
#output - a csv file with time-stamp column every 20 ms (.02s), need to figure out how to do classification of fault rows
import csv
import time
from pynput import keyboard

#--CONFIG--
NAME="FAULT_NAME.csv" #FAULTYPE_L1S_L2S_L3S 
STEP_SIZE=0.2
SAMPLE_LENGTH=20 #Total amount to save
HEADERS = ["Value1", "Value2", "Value3", "Value4"]
FAULT_MAP= {
    '1': 1,
    '2': 2,  
    '3': 3,  
    '4': 4,  
    '5': 5   
}
r1 = [230.1, 230.2, 229.8, 0.35]

#--KEYBOARD TRACKING--
current_label = 0

def on_press(key):
    global current_label
    try:
        current_label = FAULT_MAP.get(key.char, 0)
    except AttributeError:
        current_label = 0 #reset label to normal

def on_release(key):
    global current_label
    current_label = 0

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

#--Data Saving--
#createCSV
def create_csv(name, headers):
    with open(name, "x", newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    print(f"Created CSV file: {name}")

#save row to CSV
def save_row(file, data, curr_time, step_size=0.02):
    
    with open(file, 'a', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow([f"{curr_time:.3f}", *data, current_label])

    print(f"Added row: t+{curr_time} to file")

#--EXECUTION--
create_csv(NAME, HEADERS)

start_time = time.time()
elapsed = 0
index = 0

print(f"Recording {SAMPLE_LENGTH}s of data.")

try:
    while elapsed < SAMPLE_LENGTH:
        loop_start = time.time()

        save_row(NAME, r1, elapsed, current_label)
        index += 1
        elapsed = index * STEP_SIZE

        #precise 20ms of sleep if faster
        time_to_sleep = STEP_SIZE - (time.time() - loop_start)
        if time_to_sleep > 0:   
            time.sleep(time_to_sleep)

except KeyboardInterrupt:
    print("Stop triggered by user")

print("Finished Collection")
listener.stop()