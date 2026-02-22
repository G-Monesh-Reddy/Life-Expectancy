import streamlit as st
import joblib
import pandas as pd

# =========================
# Load Model Artifacts
# =========================
scaler = joblib.load("scaler.pkl")
model = joblib.load("life_expectancy_knn.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Life Expectancy Predictor")

st.title("🌍 Life Expectancy Prediction System")
st.markdown("Enter country-level health and economic indicators below.")

st.divider()

# =========================
# Input Fields with Proper Ranges & Descriptions
# =========================

st.subheader("Health Indicators")

AdultMortality = st.number_input(
    "Adult Mortality (Probability of dying between 15-60 years)",
    min_value=0,
    max_value=800,
    value=150,
    help="Typical range: 50–400. Higher values indicate poorer healthcare conditions."
)

infantdeaths = st.number_input(
    "Infant Deaths (per 1000 live births)",
    min_value=0,
    max_value=200,
    value=30,
    help="Typical range: 0–100. High-income countries usually <10."
)

HepatitisB = st.slider(
    "Hepatitis B Immunization Coverage (%)",
    0, 100, 80,
    help="Percentage of immunized population. 80–100% is considered strong coverage."
)

Polio = st.slider(
    "Polio Immunization Coverage (%)",
    0, 100, 85,
    help="Higher values indicate better vaccination programs."
)

Diphtheria = st.slider(
    "Diphtheria Immunization Coverage (%)",
    0, 100, 82,
    help="Typically above 80% in developed nations."
)

HIV = st.number_input(
    "HIV/AIDS Rate (Deaths per 1000)",
    min_value=0.0,
    max_value=50.0,
    value=0.1,
    help="Usually below 5 in most countries."
)

st.divider()

st.subheader("Economic Indicators")

GDP = st.number_input(
    "GDP per Capita (USD)",
    min_value=0.0,
    max_value=200000.0,
    value=2000.0,
    help="Typical range: 500 – 80,000 USD."
)

Population = st.number_input(
    "Population",
    min_value=1000,
    max_value=2000000000,
    value=10000000,
    help="Enter total population of the country."
)

IncomeComp = st.slider(
    "Income Composition of Resources (0–1 Index)",
    0.0, 1.0, 0.65,
    help="Closer to 1 means higher human development."
)

Schooling = st.number_input(
    "Average Years of Schooling",
    min_value=0.0,
    max_value=20.0,
    value=12.0,
    help="Developed countries usually 12–16 years."
)

st.divider()

st.subheader("Other Health Metrics")

BMI = st.number_input(
    "Average BMI",
    min_value=10.0,
    max_value=50.0,
    value=23.4,
    help="Normal BMI range: 18.5 – 24.9"
)

Alcohol = st.number_input(
    "Alcohol Consumption (Litres per capita)",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    help="Typical range: 0–15 litres."
)

TotalExp = st.number_input(
    "Total Health Expenditure (% of GDP)",
    min_value=0.0,
    max_value=20.0,
    value=5.6,
    help="Most countries: 3–12% of GDP."
)

Measles = st.number_input(
    "Reported Measles Cases",
    min_value=0,
    max_value=100000,
    value=10,
    help="Higher outbreaks indicate weaker health systems."
)

UnderFive = st.number_input(
    "Under-Five Deaths",
    min_value=0,
    max_value=300,
    value=40,
    help="High-income countries usually below 10."
)

Thin1019 = st.number_input(
    "Thinness 10–19 years (%)",
    0.0, 50.0, 12.5,
    help="Malnutrition indicator."
)

Thin59 = st.number_input(
    "Thinness 5–9 years (%)",
    0.0, 50.0, 11.2,
    help="Malnutrition indicator."
)

PercentageExp = st.number_input(
    "Health Expenditure per Capita",
    0.0, 20000.0, 150.0,
    help="Annual health spending per person."
)

Status = st.selectbox(
    "Country Development Status",
    ["Developing", "Developed"],
    help="Developed countries generally have higher life expectancy."
)

status_encoded = 1 if Status == "Developed" else 0

st.divider()

# =========================
# Prediction Button
# =========================

if st.button("Predict Life Expectancy"):

    input_dict = {
        'AdultMortality': AdultMortality,
        'infantdeaths': infantdeaths,
        'Alcohol': Alcohol,
        'percentageexpenditure': PercentageExp,
        'HepatitisB': HepatitisB,
        'Measles': Measles,
        'BMI': BMI,
        'under-fivedeaths': UnderFive,
        'Polio': Polio,
        'Totalexpenditure': TotalExp,
        'Diphtheria': Diphtheria,
        'HIV/AIDS': HIV,
        'GDP': GDP,
        'Population': Population,
        'thinness10-19years': Thin1019,
        'thinness5-9years': Thin59,
        'Incomecompositionofresources': IncomeComp,
        'Schooling': Schooling,
        'Status': status_encoded
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_order]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Life Expectancy: {round(prediction[0], 2)} years")