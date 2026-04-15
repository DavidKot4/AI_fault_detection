import sys
import os
sys.path.append(os.path.abspath(".."))

from flask import Flask, jsonify
from flask_cors import CORS
from modbus_reader import read_modbus
from model.model import predict
import time

app = Flask(__name__)
CORS(app)

@app.route("/data")
def get_data():
    try:
        raw_data = read_modbus()

        result = predict(raw_data)

        return jsonify({
            "timestamp": time.strftime("%H:%M:%S"),
            "fault_type": result["fault_type"],
            "confidence": result["confidence"],
            **raw_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)