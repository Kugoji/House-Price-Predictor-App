# Calgary House Price Predictor App 
A personal project of mine where I provide data analysis, make multiple machine learning models, and allow users to then predict what the price of a house might be in **Calgary, AB** given their own parameter input values!  
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

An interactive graph made using `plotly` shows a **scatter plot** between the `price` and `square footage` of houses within Calgary, based on their respective **area quadrant** (address). The user can interact directly with the graph by zooming in or out, and selecting datapoints. Selecting a datapoint will show additional information regarding that specific house such as the `address`, `square footage` and `number of bedrooms`.  

The user can also select which **center statistic** of the house prices they would like to view (`mean`, `median`, `mode`) and they can then see it broken down by **area quadrant**.  

Another interactive graph made using `plotly` will show the distribution of `house prices` based on their `area quadrant`, enabling users to interact directly with the histrograms to see additional information on how many houses are present within a specific price range.  

### `Model Results Mode`
This mode functions as somewhat of an "exhibit" where users can not only view, but compare the specifications and make-up of the various models that will be used to form predictions. 

The user can choose between either a `Basic OLS`, `OLS With Interaction Terms` or `Polynomial` regression model. Depending on what the user selects, the respective equation of the model will be shown, displaying the variables used. 

From this model selection, the user can then select an `output` from the model which can be in the form of:  

`Model Specs` - which shows the **specifications** and **statistical significance** of the variables used in the selected model
`VIF Test` - which shows the value of a VIF test, signalling the level of **multicollinearity** present within the model 
`Residual Plot` - which presents an **interactive scatterplot** of the residuals from the model, showing how **far off predictions are from actual observations** 









