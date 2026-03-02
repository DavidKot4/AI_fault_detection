from pymodbus.client import ModbusTcpClient
from struct import pack, unpack

def decode_register(reg1, reg2):
    raw = pack('>HH', reg1, reg2)
    return unpack('>f', raw)[0]

def poll_device(client, ip):
    
    result = client.read_holding_registers(address=1, count=50)
    
    if result.isError():
        print(f"[{ip}] Error reading registers: {result}")
        return
    
    registers = result.registers
    
    # Pair registers into 32-bit values
    pairs = [(registers[i], registers[i + 1]) 
             for i in range(0, len(registers), 2)]

    floats = [decode_register(hi, lo) for hi, lo in pairs]

    return floats


