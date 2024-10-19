import pandas as pd
import streamlit as st
from breast_cancer import (
    add_sidebar_breast, 
    get_radar_chart, 
    add_predictions, 
    get_line_chart_breast, 
    plot_diagnosis_pie_chart
)
from PIL import Image
import io

# Helper function to load CSS styling


# Sidebar for selecting the prediction type
def add_sidebar():
    st.sidebar.header("Select a Prediction Model")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ("Cancer Prediction", "Image-based Cancer Prediction")
    )
    return model_choice

# Data-based cancer prediction logic
def data_based_prediction():
    input_data = add_sidebar_breast()  # Collect input from user
    st.title("Cancer Predictor")
    
    st.info("Our data-based cancer prediction model leverages machine learning to analyze key medical features like age, tumor size, and texture, providing an accurate risk assessment for cancer.")
    
    # Display visualizations
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.subheader("Visualization")
        col3, col4 = st.columns(2)
        
        with col3:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)
            st.image('__results___40_0 (1).png', width=400)
        
        with col4:
            plot_diagnosis_pie_chart()
            st.image('__results___47_0.png')
    
    with col2:
        st.subheader("Prediction")
        add_predictions(input_data)
    
    # Display line chart
    st.subheader("Historical Data")
    line_chart = get_line_chart_breast(input_data)
    st.plotly_chart(line_chart)

    # Model performance report
    report_data = {
        'Label': ['0', '1', 'accuracy', 'macro avg', 'weighted avg'],
        'precision': [0.97, 0.93, None, 0.95, 0.96],
        'recall': [0.96, 0.95, None, 0.96, 0.96],
        'f1-score': [0.96, 0.94, 0.96, 0.95, 0.96],
        'support': [71, 43, 114, 114, 114]
    }
    
    report_df = pd.DataFrame(report_data)

    # Performance summary
    st.title('Model Performance Report')
    st.subheader('Scores')
    st.info(f"**Test Score:** 0.956")
    st.info(f"**Train Score:** 0.970")
    
    # Display classification report
    st.subheader('Classification Report')
    st.table(report_df)

# Image-based cancer prediction logic
def image_based_prediction():
    st.title("Image-Based Cancer Predictor")
    
    # User uploads an image
    uploaded_image = st.file_uploader("Upload an image for cancer prediction", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        # Convert the image to displayable format
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.subheader("Prediction")
        
        # Simulating image prediction output (this is where you'd call your model)
        st.info("Analyzing the image to predict cancer presence...")
        
        # Display a mock prediction result (replace this with actual model predictions)
        st.success("Prediction: Malignant Tumor Detected")
        
        # Display a mock activation map (replace with model’s real output)
        st.image("Breast-Mammography-Adapted-from-98.png", caption="Activation Map Highlighting Cancerous Region", use_column_width=True)

# Main function to run the app
def main():
    st.set_page_config(
        page_title="Multi_Disease_Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom styles

    # Sidebar for model selection
    model_choice = add_sidebar()
    
    if model_choice == "Cancer Prediction":
        data_based_prediction()  # Call the data-based prediction logic
    elif model_choice == "Image-based Cancer Prediction":
        image_based_prediction()  # Call the image-based prediction logic

# Run the app
if __name__ == "__main__":
    main()
