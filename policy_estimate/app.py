import streamlit as st
import pandas as pd
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)


# ---------------------------------
# Load model and scaler
# ---------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "insurance_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

    return model, scaler


model, scaler = load_model()


# ---------------------------------
# Title
# ---------------------------------
st.title("🏥 Medical Insurance Cost Predictor")

st.write(
    "Enter the patient's information below to estimate "
    "their medical insurance charges."
)


st.divider()


# ---------------------------------
# Input section
# ---------------------------------
st.subheader("Patient Information")


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )


with col2:

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    smoker = st.selectbox(
        "Smoker",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "Region",
        [
            "Northeast",
            "Northwest",
            "Southeast",
            "Southwest"
        ]
    )


# ---------------------------------
# Convert inputs
# ---------------------------------

is_female = 1 if sex == "Female" else 0

is_smoker = 1 if smoker == "Yes" else 0

region_southeast = 1 if region == "Southeast" else 0

bmi_category_obese = 1 if bmi >= 30 else 0


# ---------------------------------
# Prediction button
# ---------------------------------

st.divider()

if st.button(
    "💰 Predict Insurance Cost",
    type="primary",
    use_container_width=True
):

    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "is_female": [is_female],
        "bmi": [bmi],
        "children": [children],
        "is_smoker": [is_smoker],
        "region_southeast": [region_southeast],
        "bmi_category_obese": [bmi_category_obese]
    })


    # Scale numerical columns
    numerical_columns = [
        "age",
        "bmi",
        "children"
    ]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )


    # Make prediction
    prediction = model.predict(input_data)[0]


    # ---------------------------------
    # Display result
    # ---------------------------------

    st.success("Prediction completed successfully!")

    st.metric(
        label="Estimated Insurance Cost",
        value=f"{prediction:,.2f}"
    )


    # ---------------------------------
    # Show entered information
    # ---------------------------------

    st.subheader("Patient Details")

    display_data = pd.DataFrame({
        "Feature": [
            "Age",
            "Sex",
            "BMI",
            "Children",
            "Smoker",
            "Region"
        ],
        "Value": [
            age,
            sex,
            bmi,
            children,
            smoker,
            region
        ]
    })

    st.table(display_data)
