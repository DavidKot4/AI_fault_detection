import sys
import os
sys.path.append(os.path.abspath(".."))

from pymodbus.client import ModbusTcpClient
from modbus_polling import poll_device

MODBUS_IP = "192.168.168.11"
PORT = 502

def read_modbus():
    client = ModbusTcpClient(MODBUS_IP, port=PORT)

    if not client.connect():
        raise Exception("Could not connect to Modbus device")

    values = poll_device(client, MODBUS_IP)

    client.close()

    if values is None:
        raise Exception("Failed to read Modbus data")

    if len(values) < 15:
        raise Exception(f"Not enough Modbus data returned: {len(values)} values")

    data = {
        "V_L1": values[0],
        "V_L2": values[1],
        "V_L3": values[2],

        "V_L1_L2": values[3],
        "V_L2_L3": values[4],
        "V_L3_L1": values[5],

        "I_L1": values[6],
        "I_L2": values[7],
        "I_L3": values[8],

        "VA_L1": values[9],
        "VA_L2": values[10],
        "VA_L3": values[11],

        "W_L1": values[12],
        "W_L2": values[13],
        "W_L3": values[14],

        "Q_L1": 0,
        "Q_L2": 0,
        "Q_L3": 0,

        "PF_L1": 0,
        "PF_L2": 0,
        "PF_L3": 0,

        "THD_L1": 0,
        "THD_L2": 0,
        "THD_L3": 0,
    }

    return data