import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected successfully to mosquitto broker")

        client.subscribe('LN_data')
    else:
        print(f'Connection failed with code {reason_code}')

def on_message(client, userdata, msg):
    try:
        raw_payload = msg.payload.decode('utf-8')

        data = json.loads(raw_payload)

        print(f'Recieved data - {data}')

    except json.JSONDecodeError:
        print("Error: Received message was not valid JSON")
    except Exception as e:
        print(f'Unexpected error occured: {e}')

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

client.on_connect = on_connect
client.on_message = on_message

client.connect('localhost', 1883, 60)
print("Subscriber connected, press Ctrl+C to stop")

client.loop_forever()