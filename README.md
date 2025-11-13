# Calgary House Price Predictor App 
A personal project of mine where I provide data analysis, make multiple machine learning models, and allow users to then predict what the price of a house might be in **Calgary, AB** given their own input variable values!  

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

The user can choose between either a `Basic OLS`, `OLS With Interaction Terms` or `Polynomial (Quadratic)` regression model. Depending on what the user selects, the respective equation of the model will be shown, displaying the variables used. 

From this model selection, the user can then select an `output` from the model which can be in the form of:  

`Model Specs` - which shows the **specifications** and **statistical significance** of the variables used in the selected model  
`VIF Test` - which shows the value of a VIF test, signalling the level of **multicollinearity** present within the model   
`Residual Plot` - which presents an **interactive scatterplot** of the residuals from the chosen model, showing how **far off predictions are from actual observations**  

Since there are two columns, the user can **view two seperate models at the same time**, comparing the values of each model with another.  

### `Prediction Mode` 
This mode is where the fun happens! The user can now select a given model and use it to form their own **predictions on house prices** based on their **input variable values!**

To get started, there are important rules that must be followed so the app functions as designed: 

- **Only integer** values can be used for the `bed` variable `(eg. 3)`
- **Integer** or **.5** values can be used for the `bathroom` variable `(eg. 2.5)`
- To choose `SW` as your area quadrant, **leave all other area quadrants with the input of 0**
- **Do not select more than one area quadrant as having a value of 1**

Using `Choose model` the user can then select one of the previously named models that will be forming their house price predictions.  

Once selected, the user can choose the values of the **input variables** like the `square footage`, `number of beds`, `number of baths`.  

To select an **area quadrant**, the user must insert a value of `1` for the desired quadrant while leaving the others at `0`. To show the `SW` quadrant, the user must leave all the displayed area quadrants with a value of `0`.  

To then finally **produce the prediction**, the user must select `Intialize model` on the **left-hand side**. Once selected, a predicted price of the "configured" house will be shown!   

The different models will have different prediction values, even when given the same input values, so try to explore their differences and compare which ones might be more accurate! 

## Lessons Learned  
This was not only my first time using `Python` to create a web-app, but my first time creating even creating a web-app in the first place! There was so much learning that I had to do, most of which was during the actual coding process, but a signiciant portion of it was in the setup. I had to think of an idea that I thought would not only prove useful for myself in terms of experience, but also useful from a user perspective. I took a relevant topic of housing prices and had to visualize how I turn this abstract idea into an interactve application that someone could use. While the purpose of this app was mainly to improve my coding skills with `Python` and `streamlit`, it was also a great opportunity to showcase some elements of my statistical knowledge as well, which is why I chose to delve more into the "data scientist experience". I wanted the user to really feel like they were coming up with insights themselves just like a data scientist would and not simply punching in numbers to get a price value. There were many hiccups and learning curves, but overall making this project was such a fun experience that gave me useful outlets for my creativity and curiousity.










