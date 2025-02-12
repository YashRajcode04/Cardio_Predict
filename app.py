import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess the data
def load_data():
    try:
        # Load the dataset
        data = pd.read_csv('heart_disease.csv')

        # Ensure the required columns are present
        required_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert categorical columns to numeric
        data = data.replace({
            'Sex': {'M': 1, 'F': 0},
            'ChestPainType': {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4},
            'RestingECG': {'Normal': 1, 'ST': 2, 'LVH': 3},
            'ExerciseAngina': {'Y': 1, 'N': 0},
            'ST_Slope': {'Up': 1, 'Flat': 2, 'Down': 3}
        })

        # Convert target column to 'Yes' and 'No'
        data['HeartDisease'] = data['HeartDisease'].map({1: 'Yes', 0: 'No'})

        # Split data into features and target
        X = data.drop('HeartDisease', axis=1)
        y = data['HeartDisease']

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return data, X_scaled, y, scaler
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Train the model
def train_model(X, y):
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Streamlit app
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Header and Dark Gradient Background
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e1e, #2e2e2e);
        color: #e0e0e0;
    }
    .stApp {
        background: linear-gradient(135deg, #1e1e1e, #2e2e2e);
        color: #e0e0e0;
    }
    .title {
        text-align: center;
        font-size: 2em;
        color: #ffffff;
        margin-bottom: 20px;
    }
    .stTextInput, .stSelectbox, .stButton, .stCheckbox, .stNumberInput {
        background-color: #333333;
        color: #e0e0e0;
    }
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<p class="title">Heart Disease Prediction App</p>', unsafe_allow_html=True)

# Navigation
page = st.selectbox("Select a Page", ["Home", "Prediction Results"])

# Load data and train model
data, X, y, scaler = load_data()
if X is not None:
    model, accuracy = train_model(X, y)
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['accuracy'] = accuracy
else:
    st.write("Data is not available for training the model.")

if page == "Home":
    st.subheader("Predict Your Heart Disease Risk")

    # User inputs for prediction
    def user_input_features():
        age = st.slider("Age", 29, 77, 50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.slider("Resting Blood Pressure", 94, 200, 120)
        cholesterol = st.slider("Cholesterol", 126, 564, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Y", "N"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 71, 202, 150)
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak", -2.6, 6.2, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        # Convert inputs to appropriate formats
        input_data = {
            'Age': age,
            'Sex': 1 if sex == 'M' else 0,
            'ChestPainType': {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4}[chest_pain_type],
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': 1 if fasting_bs == 'Y' else 0,
            'RestingECG': {'Normal': 1, 'ST': 2, 'LVH': 3}[resting_ecg],
            'MaxHR': max_hr,
            'ExerciseAngina': 1 if exercise_angina == 'Y' else 0,
            'Oldpeak': oldpeak,
            'ST_Slope': {'Up': 1, 'Flat': 2, 'Down': 3}[st_slope]
        }

        return pd.DataFrame([input_data])

    # User input
    input_data = user_input_features()

    # Navigation to Prediction Results page
    if st.button("Predict"):
        input_data_scaled = st.session_state['scaler'].transform(input_data)
        prediction = st.session_state['model'].predict(input_data_scaled)
        st.session_state['prediction'] = prediction[0]
        st.session_state['input_data'] = input_data

    st.markdown("""
        **Model Accuracy**: {:.2f}
        """.format(st.session_state.get('accuracy', 'Not Available')))

elif page == "Prediction Results":
    st.subheader("Prediction Results")

    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        if prediction == 'Yes':
            st.write("**Prediction: Heart Disease**")
        else:
            st.write("**Prediction: No Heart Disease**")

        st.write("**User Input Data**")
        st.write(st.session_state['input_data'])
    else:
        st.write("No prediction available. Please go to the Home page and make a prediction.")
