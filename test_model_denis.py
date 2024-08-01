import pandas as pd
import pickle

# Load the dataset
df = pd.read_csv("data/dataset_sales_cleaned.csv")

# Load the preprocessor and model
preprocessor = pickle.load(open("data_test/preprocessor.pkl", "rb"))
model = pickle.load(open("data_test/random_forest.pkl", "rb"))

# Prepare the features and target
X = df[preprocessor.feature_names_in_]
Y = df["Price"]

# Sample a single instance for testing
test = X.sample(1)
test_index = test.index[0]  # Get the index of the sampled test data

# Display house details and prices
print(f"""
      House details: 
      {test.to_dict()}
      
      Real price: {Y[test_index]} €
      
      Estimated price: {model.predict(preprocessor.transform(test))[0]:.2f} €
      """)
