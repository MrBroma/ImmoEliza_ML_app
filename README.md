# 🏡 House Price Prediction App

Welcome to the **House Price Prediction App**, a user-friendly Streamlit application that predicts house prices based on various features. With an intuitive interface and an AI-powered backend, this app makes it easy to estimate property prices for better decision-making in real estate investments.

![Texte alternatif](assets/home-alone.gif)


## 🌟 Features

- **Interactive Interface**: Easily input property features through an elegant sidebar.
- **Real-Time Predictions**: Get instant price predictions powered by a pre-trained machine learning model.
- **Insightful Visuals**: Understand the impact of different features on house prices.
- **Responsive Design**: Optimized for both desktop and mobile devices.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7 or later installed on your machine. You will also need `pip` for package management.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MrBroma/ImmoEliza_ML_app.git
   cd immoeliza_ml_app
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
    python -m venv env
    ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
    ```

3. **Running the app**
   ```bash
    streamlit run streamlit/app.py
    ```

Visit http://localhost:8501 in your web browser to view the app.

```
immoeliza_ml_app/
│
├── models/
│   ├── preprocessor.pkl       # Pre-trained data preprocessor
│   └── random_forest.pkl      # Pre-trained Random Forest model
│
├── data/
│   └── dataset_sales_cleaned.csv   # Dataset for predictions
│
├── streamlit/
│   ├── app.py                 # Main app script
│   └── style.css              # Custom CSS styles
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```


## ✨ Usage
Input Features

To get a prediction, input the following features in the sidebar:

- Number of Bedrooms
- Number of Bathrooms
- Construction Year
- District
- Garden: Yes/No
- Garden Area (in m²)
- Kitchen Type
- Living Area (in m²)
- Locality
- Number of Facades
- Province
- Region
- Number of Rooms
- Number of Showers
- Subtype of Property
- Surface of Plot (in m²)
- Swimming Pool: Yes/No
- Terrace: Yes/No
- PEB (Performance énergétique des bâtiments)
- State of Building
- Flooding Zone
- Furnished: Yes/No

Predicting House Prices

Click the "Predict Price" button in the sidebar after inputting the features. The predicted price will be displayed instantly.

## 🤖 Model Details

The prediction model is a Random Forest Regressor trained using scikit-learn. It leverages a comprehensive dataset to provide accurate price predictions based on the input features.

## 📈 Dataset

The dataset used for training the model is dataset_sales_cleaned.csv, which includes cleaned and processed real estate data.

## 📚 Additional Information

Version: 1.0.0
Author: Loic Rouaud
https://www.linkedin.com/in/loic-rouaud/
License: MIT License

## 👥 Contributing

We welcome contributions! If you'd like to contribute, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a Pull Request.


## ⭐️ Acknowledgments

We would like to thank the contributors and open-source community for their invaluable contributions.
