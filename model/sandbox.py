import torch
import torch.nn.functional as F
import joblib
import numpy as np
from model import FaultMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running model with: {device}")

#LOAD SCALER
scaler = joblib.load('data_scaler.pkl')

#INITIALIZE & LOAD MODEL
model = FaultMLP(input_size=24, num_classes=5)
model.load_state_dict(torch.load('fault_model_v1.pth', map_location=device))
model.to(device)
model.eval()
print("Loaded model successsfully")

sample_data = [
    [97.57392883,111.976738,137.8662872,190.2674713,199.9737701,186.5434265,1.964585185,1.861226916,0.16560936,191.6922913,208.4141235,22.83194733,24.83694458,21.95410538,19.8719101,114.2391129,102.6100845,9.966773987,0.129566729,0.105338857,0.870355487,6.3297019,5.5312047,5.090535164], #2-phase-ground
    [92.38965606689452,119.06305694580078,109.48802185058594,182.7675781,183.8673553466797,185.73097229003903,3.2435262203216557,3.695946693,3.647140502929688,299.66827392578125,440.05072021484375,399.3182067871094,24.278688430786133,51.80549621582031,-92.65943909,298.1677551269531,436.4158935546875,387.9502868652344,0.081018545,0.1177261918783187,0.2320441156625747,5.825048923492432,5.475919723510742,4.5583086013793945],
    [102.8526764,128.3700714,118.9408112,201.9254456,201.998291,200.3554535,0.463716686,0.477945924,0.508413851,47.69450378,61.3539505,60.47115707,-46.65128326,-58.79013824,-59.41275787,2.09734869,12.39669514,-3.596766472,0.978127003,0.958212733,0.982497454,5.834860325,5.57796526,4.606277466] 
]


#INFERENCE LOOP
for i, raw_row in enumerate(sample_data):
    # Scale and Reshape
    scaled_data = scaler.transform(np.array(raw_row).reshape(1, -1))
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        conf, predicted = torch.max(probabilities, 1)
        percent = conf.item() * 100

    all_probs = probabilities.cpu().numpy()[0]
    print(f"Sample {i+1} Prediction: Class {predicted.item()} ({percent:.2f}% Confidence)")
    
    print(f"  Total Model Predictions: { [f'{p*100:.1f}%' for p in all_probs] }")
