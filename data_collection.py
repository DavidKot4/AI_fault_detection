from pymodbus.client import ModbusTcpClient
from struct import pack, unpack
import time
import numpy as np

device_ips = ['192.168.168.11']
duration = 0.1 * 60  
start_time = time.time()

register_log = []  # Stores batches of 30 registers

print("Starting data collection...")

try:
    while time.time() - start_time < duration:
        for ip in device_ips:
            client = ModbusTcpClient(ip, port=502)
            if client.connect():
                result = client.read_holding_registers(address=1, count=50)
                if result.isError():
                    print(f"[{ip}] Error reading registers: {result}")
                else:
                    print(f"[{ip}] Holding register values: {result.registers}")
                    register_log.append(result.registers)
                client.close()
            else:
                print(f"[{ip}] Connection failed.")
        time.sleep(.5)
except KeyboardInterrupt:
    print("Stopped by user.")

print("Decoding data...")

# Decode 32-bit floats
def decode_float(reg1, reg2):
    raw = pack('>HH', reg1, reg2)
    return unpack('>f', raw)[0]

decoded_data = []
for reading in register_log:
    pairs = [(reading[i], reading[i + 1]) for i in range(0, len(reading), 2)]
    floats = [decode_float(hi, lo) for hi, lo in pairs]
    decoded_data.append(floats)
    print(floats)
    print(" ")

# Convert to list
data = np.array(decoded_data)
payload = data.tolist()
