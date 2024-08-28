import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the doctor notes and patient records dataset
doctor_notes = pd.read_csv('doctor_notes.csv')
doctor_notes.set_index('Patient ID', inplace=True)
doctor_notes.sort_index(inplace=True)

patient_records = pd.read_csv('patient_records.csv')
patient_records.drop('Unnamed: 0', axis = 1, inplace=True)
patient_records.set_index('Patient ID', inplace=True)
patient_records.sort_index(inplace=True)

# Preprocess the data
X_notes = doctor_notes['Doctor Notes']
y_surgery = patient_records['Surgery Type']

# Split the data
X_train_notes, X_test_notes, y_train_notes, y_test_notes = train_test_split(X_notes, y_surgery, test_size=0.2, random_state=42)

# Create a text classification pipeline
text_clf = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
text_clf.fit(X_train_notes, y_train_notes)

# Evaluate the model
y_pred_notes = text_clf.predict(X_test_notes)



# Handle 'Not determined' values in the Outcome column
patient_records['Outcome'] = patient_records['Outcome'].replace('Not determined', np.nan)

# Drop rows with NaN values in the Outcome column
patient_records_encoded = patient_records.dropna(subset=['Outcome'])


# Preprocess patient records
label_encoder = {}
categorical_columns = ['Gender', 'Outcome', 'Surgery Type']  # Add other categorical columns as needed
for col in categorical_columns:
    le = LabelEncoder()
    patient_records_encoded[col] = le.fit_transform(patient_records_encoded[col])
    label_encoder[col] = le

# One-Hot Encoding for 'Surgery Type'
# patient_records_encoded = pd.get_dummies(patient_records_cleaned, columns=['Surgery Type'], drop_first=True)

# Define features and target for the determined dataset
X_determined = patient_records_encoded.drop('Outcome', axis=1)
y_determined = patient_records_encoded['Outcome']

# Train the Random Forest model (assuming this is the final model)
X_train, X_test, y_train, y_test = train_test_split(X_determined, y_determined, test_size=0.2, random_state=42)
# Initialize the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(X_train_scaled, y_train)


# Prepare the input for the model
input_data = {}

# Streamlit Dashboard
st.title('Keratoplasty Outcome Prediction')
# Input for doctor notes
notes_input = st.text_area('Enter Doctor Notes:', '')

# Input for other features
age = st.slider('Age:', min_value=0, max_value=120)

gender = st.selectbox('Gender:', options=patient_records['Gender'].unique().tolist())

days_last_visit = (datetime.now().date() - st.date_input('Surgery Date:', value=datetime.today())).days

# Predict the surgery type from the notes
surgery_pred = text_clf.predict([notes_input])[0]

input_data = {
    'Surgery Type': surgery_pred,
    'Age': age,
    'Gender': gender,
    'Days since last visit': days_last_visit,
}

# Display prediction
st.write("Prediction for the input:")
st.write(pd.DataFrame([input_data]))
st.write(f'Predicted Surgery Type: {surgery_pred}')
days_last_visit = input_data['Days since last visit']
st.write(f'Days Since Last Visit: {days_last_visit}')

# One-hot encode the Surgery Type
input_data['Gender'] = label_encoder['Gender'].transform([gender])[0]
input_data['Surgery Type'] = label_encoder['Surgery Type'].transform([surgery_pred])[0]

input_df = pd.DataFrame([input_data])

# Predict the outcome
input_df_scaled = scaler.transform(input_df)
outcome_pred = logistic_regression.predict(input_df_scaled)[0]
outcome_pred = label_encoder['Outcome'].inverse_transform([outcome_pred])[0]

st.write(f'Predicted Outcome: {outcome_pred}')

