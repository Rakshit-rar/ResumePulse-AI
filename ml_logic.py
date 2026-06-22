
import pickle
import pandas as pd

def predict_qualification(skills_found, all_features):
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Create feature vector
    input_data = pd.DataFrame([{feature: (1 if feature in skills_found else 0) for feature in all_features}])
    prediction = model.predict(input_data)
    return "Qualified" if prediction[0] == 1 else "Needs More Skills"
