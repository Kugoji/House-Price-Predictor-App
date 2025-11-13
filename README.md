# Calgary House Price Predictor App 
A personal project of mine where I provide data analysis, make multiple machine learning models, and allow users to then predict what the price of a house might be in **Calgary, AB** given their own parameter choices!  
Here's the link to interact with the app directly: https://yyc-house-prices.streamlit.app 

## How It's Made
**Tech & Libraries used:** `Python`, `Streamlit`, `Pandas`, `Numpy`, `Statsmodels`, `Scikit-learn`, `Plotly`

To make this app, I did the coding in `Python`, utilizing the `streamlit` library in order to design the front-end. This app uses an expansive kaggle dataset that contains information on houses within **Calgary**. The data was pulled from various real-estate websites in **2023** and as such, the prices present in the dataset may be slightly outdated. Also, while expansive, the dataset does **not** contain the information of **all** house prices in Calgary, only houses found on **public listing sites** will be present.   

To use this app, there are **3 modes** the user can select from:  

`EDA Mode` - Where the user can **explore** and **visualize** the data present in the dataset  
`Model Results Mode` - Shows the **specs** of multiple models and allows the user to **compare key metrics**  
`Prediction Mode` - Allows the user to select a model and make **predictions based on key factors**  

### `EDA Mode`  
**Exploratory Data Analysis** mode allows the user to interact with the data in a more analytical sense, providing comprehensive information about the house prices present within the dataset.   

The user can view the values of key **summary statistics** such as the `mean`,`median`, `min`, `max`, `quartiles` and `standard deviation`. 


