
import paho.mqtt.client as mqtt
import json
import time

BROKER_ADDRESS="localhost"
PORT=1883
TOPIC='LN_data'

#create client object w/ latest library
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

#Attempt broker connection w/ client
try:
    client.connect(BROKER_ADDRESS, PORT)
    print(f"Connected to local broker at {BROKER_ADDRESS}")
except Exception as e:
    print(f'Error in connecting to local broker: {e}')
    exit()

#Attempt to send data
try:
    while True:
       payload = {
           "Hello" : "World!"
       } 
       
       json_data = json.dumps(payload)
       
       client.publish(TOPIC, json_data)

       print(f'Published to broker {json_data}')
       
       time.sleep(5)
       
except KeyboardInterrupt:
    print('Disconnecting...')
    client.disconnect()

