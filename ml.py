import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import os
import base64

# This must be the first Streamlit command
st.set_page_config(page_title="Autism Prediction App", page_icon="🌟", layout="wide")

# Load the saved model and scaler using local file paths
model_path = 'entire_model.pkl'
model_data = joblib.load(model_path)
voting_classifier = model_data['model']
scaler = model_data['scaler']

def get_base64_image():
    # Try different image paths
    for ext in ["png", "jpg", "jpeg", "webp"]:
        image_path = f"background.{ext}"  # Changed to match your filename
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    return None

bg_img = get_base64_image()

# --- Custom CSS with Adjusted Background Opacity ---
custom_css = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.45), rgba(255, 255, 255, 0.85)),
                    url("data:image/png;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.95) !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #2c3e50 !important;
    }}
    .stMarkdown p {{
        color: #333333 !important;
    }}
    </style>
"""

if bg_img:
    st.markdown(custom_css, unsafe_allow_html=True)
else:
    st.warning("Background image not found. Using plain background.")
    st.markdown("""
    <style>
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.95);
    }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation (with buttons)
def navigation_buttons():
    st.sidebar.title("Navigation")
    
    # Navigation buttons to select page with icons
    if st.sidebar.button("🏠 Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("🔮 Predict"):
        st.session_state.page = "Predict"
    if st.sidebar.button("📊 Dashboard"):
        st.session_state.page = "Dashboard"

# Initialize session state for page
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Display navigation buttons
navigation_buttons()

# Home Page (Centered content)
if st.session_state.page == "Home":
    st.title("Welcome to the Autism Prediction App 🌍", anchor="center")

    st.markdown(
        """
        <div style="text-align: center;">
            ### About Autism Spectrum Disorder (ASD):
            Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication, behavior, and social interactions. This app uses machine learning to assist in early prediction of ASD.

            #### Let's make early detection accessible and impactful!
        </div>
        """, unsafe_allow_html=True
    )

    # Using Streamlit icons for animations
    st.markdown(
        """
        <div style="text-align: center;">
            <i class="fa fa-smile-o" style="font-size: 100px;"></i> 
            <p>Join us in spreading awareness for ASD!</p>
        </div>
        """, unsafe_allow_html=True
    )

# Predict Page
elif st.session_state.page == "Predict":
    st.title("Autism Prediction 🌟")

    # Collect user input
    st.markdown("### Enter the details for prediction:")

    feature_names = [
        "Social_Responsiveness_Scale", "Age_Years",
        "Speech Delay/Language Disorder", "Learning disorder",
        "Genetic_Disorders", "Depression",
        "Global developoental delay/intellectual disability",
        "Social/Behavioural Issues", "Anxiety_disorder",
        "Sex", "Jaundice", "Family_member_with_ASD"
    ]

    categorical_columns = [
        "Speech Delay/Language Disorder", "Learning disorder",
        "Genetic_Disorders", "Depression",
        "Global developoental delay/intellectual disability",
        "Social/Behavioural Issues", "Anxiety_disorder",
        "Sex", "Jaundice", "Family_member_with_ASD"
    ]

    user_input = []
    for col in feature_names:
        if col in categorical_columns:
            if col == "Sex":
                value = st.selectbox(f"{col}:", ["Male", "Female"])
                user_input.append(1 if value == "Male" else 0)
            else:
                value = st.selectbox(f"{col}:", ["Yes", "No"])
                user_input.append(1 if value == "Yes" else 0)
        else:
            value = st.number_input(f"{col}:", step=0.1)
            user_input.append(value)

    if st.button("🔮 Predict", key="predict_button_in_predict_page"):
        # Process input and make predictions
        user_input_array = np.array(user_input).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input_array)
        prediction = voting_classifier.predict(user_input_scaled)

        st.markdown("### Prediction Result:")
        if prediction[0] == 1:
            st.success("The individual is likely to have Autism Spectrum Disorder.")
        else:
            st.success("The individual is unlikely to have Autism Spectrum Disorder.")

        # Icon to indicate prediction success
        st.markdown(
            """
            <div style="text-align: center;">
                <i class="fa fa-thumbs-up" style="font-size: 100px; color: green;"></i>
            </div>
            """, unsafe_allow_html=True
        )

# Dashboard Page (Dynamic Visualizations)
elif st.session_state.page == "Dashboard":
    st.title("Dashboard 📊")

    st.markdown("### Explore relationships between features:")

    # Load dataset using the local file path
    dataset_path = r'C:\Users\mohammed bilal\OneDrive\Desktop\ISL\expanded_asd_data.csv'
    autism_dataset = pd.read_csv(dataset_path)

    # Encode categorical variables for visualization
    categorical_columns = [
        "Speech Delay/Language Disorder", "Learning disorder",
        "Genetic_Disorders", "Depression",
        "Global developoental delay/intellectual disability",
        "Social/Behavioural Issues", "Anxiety_disorder",
        "Sex", "Jaundice", "Family_member_with_ASD"
    ]

    for col in categorical_columns:
        le = LabelEncoder()
        autism_dataset[col] = le.fit_transform(autism_dataset[col])

    # User selects features to visualize
    x_feature = st.selectbox("Select the X-axis feature:", autism_dataset.columns)
    y_feature = st.selectbox("Select the Y-axis feature:", autism_dataset.columns)

    # Create a dynamic plot (Line, Bar, or Scatter)
    plot_type = st.selectbox("Select plot type:", ["Line", "Bar", "Scatter"])

    if st.button("📊 Generate Visualization", key="generate_visualization_button"):
        if plot_type == "Line":
            fig = px.line(
                autism_dataset, x=x_feature, y=y_feature, title=f"Line plot: {x_feature} vs {y_feature}"
            )
        elif plot_type == "Bar":
            fig = px.bar(
                autism_dataset, x=x_feature, y=y_feature, title=f"Bar plot: {x_feature} vs {y_feature}"
            )
        elif plot_type == "Scatter":
            fig = px.scatter(
                autism_dataset, x=x_feature, y=y_feature, title=f"Scatter plot: {x_feature} vs {y_feature}"
            )

        st.plotly_chart(fig)

    st.markdown(
        """
        ### Insights:
        - Use these visualizations to analyze relationships between features and outcomes.
        """
    )
