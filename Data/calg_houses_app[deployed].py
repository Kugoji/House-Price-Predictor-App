# Goal will be to produce an app using calgary housing data to see how prices differ based on other factors
# There will be 3 modes: 
# EDA (explanatory data analysis): shows visualizations and summary statistics of the house prices for each quadrant
# Regression results: shows the coefficients of the variables chosen and other useful information like SER and R squared 
# Prediction mode: users can put in their own values for the input variables and the model will try to predict 


#Importing packages
import pandas as pd
import numpy as np
import streamlit as st
import statsmodels as sm 
import statsmodels.api as smi
import seaborn as sb 
import matplotlib as mpl
import plotly as plot
import plotly.express as px 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score 
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

#Preprocessing Previous Data
BASE_DIR = Path(__file__).resolve().parent  # folder containing this script, e.g., Data/
data_path = BASE_DIR / "yyc_house_prices.xlsx"
logo_path = BASE_DIR / "logo.png" 
image_path = BASE_DIR / "house_image.png"

@st.cache_data
def load_houses():
    return pd.read_excel(data_path)

#First load dataset
houses = load_houses()

#Cleaning data

houses_clean = houses.dropna()
#Removes houses with 0 bedrooms and 0 bathrooms
houses_clean = houses_clean.loc[~((houses_clean['Beds'] == 0) & (houses_clean['Bath'] == 0))]
#Mutating Address column so that the values just display the quadrant
houses_clean.loc[houses_clean['Address'].str.contains(r'\bSE\b', regex = True), 'Address'] = 'SE'
houses_clean.loc[houses_clean['Address'].str.contains(r'\bSe\b', regex = True), 'Address'] = 'SE'
houses_clean.loc[houses_clean['Address'].str.contains(r'\bSW\b', regex = True), 'Address'] = 'SW'
houses_clean.loc[houses_clean['Address'].str.contains(r'\bSw\b', regex = True), 'Address'] = 'SW'
houses_clean.loc[houses_clean['Address'].str.contains(r'\bNE\b', regex = True), 'Address'] = 'NE'
houses_clean.loc[houses_clean['Address'].str.contains(r'\bNe\b', regex = True), 'Address'] = 'NE'
houses_clean.loc[houses_clean['Address'].str.contains(r'\bNW\b', regex = True), 'Address'] = 'NW'
houses_clean.loc[houses_clean['Address'].str.contains(r'\bNw\b', regex = True), 'Address'] = 'NW'
#Removes houses that don't have a quadrant in their address 
houses_clean_final = houses_clean.loc[(houses_clean['Address'] == 'SE') |  (houses_clean['Address'] == 'SW') | (houses_clean['Address'] == 'NE') | (houses_clean['Address'] == 'NW')]
#Convert Address to categorical data 
houses_clean_final['Address'] = pd.Categorical(houses_clean_final['Address'], categories=['SW', 'SE', 'NW', 'NE'], ordered= True)


#Remove visible outliers
houses_clean_final = houses_clean_final.loc[~((houses_clean_final['Price'] == 1695000) & (houses_clean_final['Beds'] == 4) & (houses_clean_final['Address'] == 'NW') & (houses_clean_final['Sq.Ft'] == 39654))]
q1 = houses_clean_final[['Price']].quantile(0.25)
q3 = houses_clean_final[['Price']].quantile(0.75)
iqr = q3 - q1

#For linear model: 
houses_clean_final_linear = houses_clean_final.loc[houses_clean_final['Price'] < 6000000]

#For Polynomial model: 
houses_clean_final_poly = houses_clean_final

#Building Basic Linear Model 
X = pd.get_dummies(houses_clean_final_linear[['Address', 'Sq.Ft', 'Beds', 'Bath']], drop_first= True)
X = X.astype(float)
Y = houses_clean_final_linear[['Price']] 
model_linear = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 45)
model_linear.fit(X_train, Y_train)
Y_pred = model_linear.predict(X_test)
Y_pred = Y_pred.squeeze()
Y_array = Y_test[['Price']].squeeze().values
residuals = Y_array - Y_pred 


#Building Linear Model With Interaction Terms
X2 = pd.get_dummies(houses_clean_final_linear[['Address','Sq.Ft', 'Beds', 'Bath']], drop_first= True)
X2 = X2.astype(float)
X2['SE_x_Sq.Ft'] = X2['Address_SE'] * X2['Sq.Ft']
X2['NW_x_Sq.Ft'] = X2['Address_NW'] * X2['Sq.Ft']
X2['NE_x_Sq.Ft'] = X2['Address_NE'] * X2['Sq.Ft']
X2['SE_x_Beds'] = X2['Address_SE'] * X2['Beds']
X2['NW_x_Beds'] = X2['Address_NW'] * X2['Beds']
X2['NE_x_Beds'] = X2['Address_NE'] * X2['Beds']
Y = houses_clean_final_linear[['Price']] 
model_interac = LinearRegression()
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y, test_size = 0.2, random_state = 45)
model_interac.fit(X2_train, Y2_train)
Y_pred2 = model_interac.predict(X2_test)
Y_pred2 = Y_pred2.squeeze()
Y_array2 = Y2_test[['Price']].squeeze().values
residuals2 = Y_array2 - Y_pred2 

#Building Polynomial Model 
poly =(PolynomialFeatures(degree = 2, include_bias = False))
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(input_features= X.columns)
X_poly_df = pd.DataFrame(X_poly, columns= feature_names) 
X_poly_train, X_poly_test, Y_poly_train, Y_poly_test = train_test_split(X_poly_df, Y, test_size = 0.2, random_state = 45)
model_poly = LinearRegression()
model_poly.fit(X_poly_train, Y_poly_train) 
Y_pred_poly = model_poly.predict(X_poly_test)
Y_pred_poly = Y_pred_poly.squeeze()
Y_array_poly = Y_poly_test[['Price']].squeeze().values
residuals_poly = Y_array_poly - Y_pred_poly 

#Building Model in StatsModel

#For Linear Model
X_sm = X.astype(float)
X_sm = smi.add_constant(X) 
stat_model_linear = smi.OLS(Y, X_sm).fit()
robust_linear = stat_model_linear.get_robustcov_results(cov_type = 'HC1')

#Linear with Interaction Terms
X2_sm = X2.astype(float)
X2_sm = smi.add_constant(X2)
X2_sm = X2_sm.reset_index(drop = True)
Y = Y.reset_index(drop = True)
stat_model_interac = smi.OLS(Y,X2_sm).fit()
robust_interac = stat_model_interac.get_robustcov_results(cov_type = 'HC1')

#Polynomial Model
X_poly_sm = smi.add_constant(X_poly_df)
stat_model_poly = smi.OLS(Y, X_poly_sm).fit()
robust_poly = stat_model_poly.get_robustcov_results(cov_type = 'HC1')


#Summary Of Results
sum1 = summary_col([robust_linear], stars=True, float_format="%.4f", model_names=["OLS"])
sum2 = summary_col([robust_interac], stars=True, float_format="%.4f", model_names=["OLS"])
sum3 = summary_col([robust_poly], stars=True, float_format="%.4f", model_names=["OLS"])


# Visualizations from data using Plotly 
#EDA Plots
fig1 = px.scatter(houses_clean_final_linear, x = 'Sq.Ft', y = 'Price', color = 'Address', size = 'Beds', title = 'House Price Based on Square Footage')
fig2 = px.histogram(houses_clean_final_linear, x = 'Price', facet_col= 'Address', nbins= 50, title = 'House Price Distribution Based on Area Quadrant')
fig2.update_traces(marker_line_width = 0.2, marker_line_color = 'black')
fig3 = px.box(houses_clean_final_linear, x = 'Address', y = 'Price')

#Residual Plots
fig4 = px.scatter(x = Y_pred, y = residuals, labels= {"x": "Predicted Values", "y": "Residual"}, title= "Basic OLS Residual Plot") 
fig4.add_shape(
    type="line",
    x0=min(Y_pred),
    x1=max(Y_pred),
    y0=0,
    y1=0,
    line=dict(color="red", dash="dash"))  

fig5 = px.scatter(x = Y_pred2, y = residuals2, labels= {"x": "Predicted Values", "y": "Residual"}, title= "OLS With Interaction Terms Residual Plot") 
fig5.add_shape(
    type="line",
    x0=min(Y_pred2),
    x1=max(Y_pred2),
    y0=0,
    y1=0,
    line=dict(color="red", dash="dash")) 
fig6 = px.scatter(x = Y_pred_poly, y = residuals_poly, labels= {"x": "Predicted Values", "y": "Residual"}, title= "Polynomial (Quadratic) Residual Plot") 
fig6.add_shape(
    type="line",
    x0=min(Y_pred_poly),
    x1=max(Y_pred_poly),
    y0=0,
    y1=0,
    line=dict(color="red", dash="dash"))


#Title & Welcome Page

st.sidebar.image(logo_path)
mode = st.sidebar.selectbox("Select Mode", ["Home", "EDA Mode", "Model Results Mode", "Prediction Mode"],
                            index = 0)

if mode == "Home": 
    st.markdown("<h1 style = 'text-align: center;'> Exploring Calgary House Prices Through Regression Analysis</h1>",
                unsafe_allow_html= True) 
    st.markdown("<h5 style = 'text-align: center;'> Made by Kashie Ugoji</h5>", 
            unsafe_allow_html= True) 
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2: 
        st.image(image_path, width = 600)
    with st.expander("Background Information"):
                     st.markdown("For this analysis, we will be exploring a dataset on Kaggle composed of different houses found in **Calgary, Alberta.**"
                     " The dataset will have the variables for:")
                     st.markdown("""
                                 - **Price (CAD)** - Price of the house
                                 - **Square Footage** -  Square footage of the house
                                 - **Number of beds**  - The amount of bedrooms the house has
                                 - **Number of baths** - The amount of bathrooms the house has
                                 - **Area Quadrant (SW, SE, NW, NE)** - The area the house is located 
                                 """)
                     st.markdown("By selecting a mode, you will be able to:") 
                     st.markdown("""
                                 - view the **characteristics of the data** 
                                 - view the **results of the various regression models** 
                                 - make your own **price predictions based on the model and input variable values**
                                 """)
    st.markdown("<h2 style = 'text-align: center;'> Let's See What Results You Find!</h2>", unsafe_allow_html= True)
                    




#EDA Mode
if mode == "EDA Mode":
    st.markdown("# Exploratory Data Analysis (EDA)")
    col4, col5 = st.columns([1,1.3], border= True)
    col6, col7 = st.columns([1,1.3], border = True) 
    with col4:
        st.markdown("## Summary Stats")
        st.markdown("#### Min. - $141,900")
        st.markdown("#### 1st Qu. - $397,250")
        st.markdown("#### Mean - $729,739")
        st.markdown("#### Median - $615,195")
        st.markdown("#### 3rd Qu. - $849,900")
        st.markdown("#### Max. - $4,998,000")
        st.markdown("#### Std. Dev - $527,223")
    with col5:
          st.plotly_chart(fig1)
    with col6:
        center = st.selectbox("Choose your center statistic", ['Mean', 'Median', 'Mode'])
        if center == 'Mean':
            st.markdown("## Quadrant Means")
            st.markdown("### SW - $818,367")
            st.markdown("### SE - $637,092")
            st.markdown("### NW - $799,467")
            st.markdown("### NE - $579,509")
        elif center == 'Median':
            st.markdown("## Quadrant Medians")
            st.markdown("### SW - $649,900")
            st.markdown("### SE - $555,000")
            st.markdown("### NW - $749,000")
            st.markdown("### NE - $549,900")
        else:
            st.markdown("## Quadrant Modes")
            st.markdown("### SW - $349,900")
            st.markdown("### SE - $749,900")
            st.markdown("### NW - $849,900")
            st.markdown("### NE - $599,900")
             
    with col7:
          st.plotly_chart(fig2)
          
          
       



#Model Results Mode
if mode == "Model Results Mode":
    st.markdown("# Model Results")    
    col8, col9 = st.columns(2, border= True)
    col10, col11 = st.columns(2, border= True)
    with st.container():
        with col8:
            model_choice1 = st.selectbox("Choose Regression Model", ['Basic OLS', 'OLS With Interaction Terms', 'Polynomial (Quadratic)'], key = "model_choice1")
            if model_choice1 == "Basic OLS":
                      st.markdown(
                           r"$\text{House Price} = \beta_0 + \beta_1 \cdot \text{SqFt} + \beta_2 \cdot \text{Beds} + \beta_3 \cdot \text{Baths} + \beta_4 \cdot \text{SE} + \beta_5 \cdot \text{NW} + \beta_6 \cdot \text{NE} + \varepsilon$")
            elif model_choice1 == "OLS With Interaction Terms":
                 st.markdown(
    r"$\text{House Price} = \beta_0 + \beta_1 \cdot \text{SqFt} + \beta_2 \cdot \text{Beds} + \beta_3 \cdot \text{Baths} + \beta_4 \cdot \text{SE} + \beta_5 \cdot \text{NW} + \beta_6 \cdot \text{NE} + \beta_7 \cdot (\text{SE} \cdot \text{SqFt}) + \beta_8 \cdot (\text{NW} \cdot \text{SqFt}) + \beta_9 \cdot (\text{NE} \cdot \text{SqFt}) + \beta_{10} \cdot (\text{SE} \cdot \text{Beds}) + \beta_{11} \cdot (\text{NW} \cdot \text{Beds}) + \beta_{12} \cdot (\text{NE} \cdot \text{Beds}) + \varepsilon$")
            elif model_choice1 == "Polynomial (Quadratic)":
                 st.markdown(r"""
$\text{House Price} = \beta_0 + \beta_1 \cdot \text{SqFt} + \beta_2 \cdot \text{Beds} + \beta_3 \cdot \text{Baths} + \beta_4 \cdot \text{SE} + \beta_5 \cdot \text{NW} + \beta_6 \cdot \text{NE}$
<br>
$+ \beta_7 \cdot \text{SqFt}^2 + \beta_8 \cdot \text{Beds}^2 + \beta_9 \cdot \text{Baths}^2$
<br>
$+ \beta_{10} \cdot (\text{SqFt} \cdot \text{Beds}) + \beta_{11} \cdot (\text{SqFt} \cdot \text{Baths}) + \beta_{12} \cdot (\text{Beds} \cdot \text{Baths})$
<br>
$+ \beta_{13} \cdot (\text{SE} \cdot \text{SqFt}) + \beta_{14} \cdot (\text{NW} \cdot \text{SqFt}) + \beta_{15} \cdot (\text{NE} \cdot \text{SqFt})$
<br>
$+ \beta_{16} \cdot (\text{SE} \cdot \text{Beds}) + \beta_{17} \cdot (\text{NW} \cdot \text{Beds}) + \beta_{18} \cdot (\text{NE} \cdot \text{Beds})$
<br>
$+ \beta_{19} \cdot (\text{SE} \cdot \text{Baths}) + \beta_{20} \cdot (\text{NW} \cdot \text{Baths}) + \beta_{21} \cdot (\text{NE} \cdot \text{Baths}) + \varepsilon$
""", unsafe_allow_html=True)    
        with col10:
                output_choice1 = st.selectbox("Choose Ouput", ['Model Specs', 'VIF Test', 'Residual Plot'], key= "output_choice1")
                if model_choice1 == "Basic OLS" and output_choice1 == "Model Specs":
                    st.markdown(sum1.as_html(), unsafe_allow_html= True)
                elif model_choice1 == "Basic OLS" and output_choice1 == "VIF Test":
                    st.header("2.79 (Sq.Ft)")
                elif model_choice1 == "Basic OLS" and output_choice1 == "Residual Plot":
                     st.plotly_chart(fig4, key = "fig4")
                     st.caption("**RMSE: $272, 522**")
                elif model_choice1 == "OLS With Interaction Terms" and output_choice1 == "Model Specs":
                     st.markdown(sum2.as_html(), unsafe_allow_html= True)
                elif model_choice1 == "OLS With Interaction Terms" and output_choice1 == "VIF Test":
                    st.header("4.86 (Sq.Ft)")
                elif model_choice1 == "OLS With Interaction Terms" and output_choice1 == "Residual Plot":
                     st.plotly_chart(fig5, key = "fig5")
                     st.caption("**RMSE: $264, 414**")
                elif model_choice1 == "Polynomial (Quadratic)" and output_choice1 == "Model Specs":
                     st.markdown(sum3.as_html(), unsafe_allow_html= True)
                elif model_choice1 == "Polynomial (Quadratic)" and output_choice1 == "VIF Test":
                     st.header("36.44 (Sq.Ft)")
                elif model_choice1 == "Polynomial (Quadratic)" and output_choice1 == "Residual Plot":
                     st.plotly_chart(fig6, key= "fig5")
                     st.caption("**RMSE: $252, 490**")
    with st.container():
        with col9:
            model_choice2 = st.selectbox("Choose Regression Model", ['Basic OLS', 'OLS With Interaction Terms', 'Polynomial (Quadratic)'], key = "model_choice2")
            if model_choice2 == "Basic OLS":
                st.markdown(
                     r"$\text{House Price} = \beta_0 + \beta_1 \cdot \text{SqFt} + \beta_2 \cdot \text{Beds} + \beta_3 \cdot \text{Baths} + \beta_4 \cdot \text{SE} + \beta_5 \cdot \text{NW} + \beta_6 \cdot \text{NE} + \varepsilon$")   
            elif model_choice2 == "OLS With Interaction Terms":
                st.markdown(
        r"$\text{House Price} = \beta_0 + \beta_1 \cdot \text{SqFt} + \beta_2 \cdot \text{Beds} + \beta_3 \cdot \text{Baths} + \beta_4 \cdot \text{SE} + \beta_5 \cdot \text{NW} + \beta_6 \cdot \text{NE} + \beta_7 \cdot (\text{SE} \cdot \text{SqFt}) + \beta_8 \cdot (\text{NW} \cdot \text{SqFt}) + \beta_9 \cdot (\text{NE} \cdot \text{SqFt}) + \beta_{10} \cdot (\text{SE} \cdot \text{Beds}) + \beta_{11} \cdot (\text{NW} \cdot \text{Beds}) + \beta_{12} \cdot (\text{NE} \cdot \text{Beds}) + \varepsilon$")
            elif model_choice2 == "Polynomial (Quadratic)":
                 st.markdown(r"""
$\text{House Price} = \beta_0 + \beta_1 \cdot \text{SqFt} + \beta_2 \cdot \text{Beds} + \beta_3 \cdot \text{Baths} + \beta_4 \cdot \text{SE} + \beta_5 \cdot \text{NW} + \beta_6 \cdot \text{NE}$
<br>
$+ \beta_7 \cdot \text{SqFt}^2 + \beta_8 \cdot \text{Beds}^2 + \beta_9 \cdot \text{Baths}^2$
<br>
$+ \beta_{10} \cdot (\text{SqFt} \cdot \text{Beds}) + \beta_{11} \cdot (\text{SqFt} \cdot \text{Baths}) + \beta_{12} \cdot (\text{Beds} \cdot \text{Baths})$
<br>
$+ \beta_{13} \cdot (\text{SE} \cdot \text{SqFt}) + \beta_{14} \cdot (\text{NW} \cdot \text{SqFt}) + \beta_{15} \cdot (\text{NE} \cdot \text{SqFt})$
<br>
$+ \beta_{16} \cdot (\text{SE} \cdot \text{Beds}) + \beta_{17} \cdot (\text{NW} \cdot \text{Beds}) + \beta_{18} \cdot (\text{NE} \cdot \text{Beds})$
<br>
$+ \beta_{19} \cdot (\text{SE} \cdot \text{Baths}) + \beta_{20} \cdot (\text{NW} \cdot \text{Baths}) + \beta_{21} \cdot (\text{NE} \cdot \text{Baths}) + \varepsilon$
""", unsafe_allow_html=True)
                    
        with col11:
            output_choice2 = st.selectbox("Choose Ouput", ['Model Specs', 'VIF Test', 'Residual Plot'], key = "output_choice2")
            if model_choice2 == "Basic OLS" and output_choice2 == "Model Specs":
                st.markdown(sum1.as_html(), unsafe_allow_html= True)
            elif model_choice2 == "Basic OLS" and output_choice2 == "VIF Test":
                st.header("2.79 (Sq.Ft)")
            elif model_choice2 == "Basic OLS" and output_choice2 == "Residual Plot":
                st.plotly_chart(fig4)
                st.caption("**RMSE: $272, 522**")
            elif model_choice2 == "OLS With Interaction Terms" and output_choice2 == "Model Specs":
                st.markdown(sum2.as_html(), unsafe_allow_html= True)
            elif model_choice2 == "OLS With Interaction Terms" and output_choice2 == "VIF Test":
                st.header("4.86 (Sq.Ft)")
            elif model_choice2 == "OLS With Interaction Terms" and output_choice2 == "Residual Plot":
                st.plotly_chart(fig5)
                st.caption("**RMSE: $264, 414**")
            elif model_choice2 == "Polynomial (Quadratic)" and output_choice2 == "Model Specs":
                st.markdown(sum3.as_html(), unsafe_allow_html= True)
            elif model_choice2 == "Polynomial (Quadratic)" and output_choice2 == "VIF Test":
                st.header("36.44 (Sq.Ft)")
            elif model_choice2 == "Polynomial (Quadratic)" and output_choice2 == "Residual Plot":
                st.plotly_chart(fig6)
                st.caption("**RMSE: $252, 490**")
        
#Helper Functions For Prediction
def predict_linear(sq_ft: float, beds: int, baths: float, add_se: int, add_nw: int, add_ne: int):
     df = pd.DataFrame({
          'Sq.Ft': [sq_ft],
          'Beds': [beds],
          'Bath': [baths],
          'Address_SE': [add_se],
          'Address_NW': [add_nw],
          'Address_NE': [add_ne]
     })
     pred = model_linear.predict(df)
     squeeze = pred.squeeze()
     price = str(round(squeeze.item()))
     return "#### $" + price

def predict_interac(sq_ft: float, beds: int, baths: float, add_se: int, add_nw: int, add_ne: int):
     df = pd.DataFrame({
          'Sq.Ft': [sq_ft],
          'Beds': [beds],
          'Bath': [baths],
          'Address_SE': [add_se],
          'Address_NW': [add_nw],
          'Address_NE': [add_ne]
     })
     df['SE_x_Sq.Ft'] = df['Address_SE'] * df['Sq.Ft']
     df['NW_x_Sq.Ft'] = df['Address_NW'] * df['Sq.Ft']
     df['NE_x_Sq.Ft'] = df['Address_NE'] * df['Sq.Ft']
     df['SE_x_Beds'] = df['Address_SE'] * df['Beds']
     df['NW_x_Beds'] = df['Address_NW'] * df['Beds']
     df['NE_x_Beds'] = df['Address_NE'] * df['Beds']
     pred = model_interac.predict(df)
     squeeze = pred.squeeze()
     price = str(round(squeeze.item()))
     return "#### $" + price

def predict_poly(sq_ft: float, beds: int, baths: float, add_se: int, add_nw: int, add_ne: int):
     df = pd.DataFrame({
          'Sq.Ft': [sq_ft],
          'Beds': [beds],
          'Bath': [baths],
          'Address_SE': [add_se],
          'Address_NW': [add_nw],
          'Address_NE': [add_ne]
     })
     df_poly = poly.fit_transform(df)
     column_names = poly.get_feature_names_out(input_features= df.columns)
     new_df = pd.DataFrame(df_poly, columns= column_names)
     pred = model_poly.predict(new_df)
     squeeze = pred.squeeze()
     price = str(round(squeeze.item()))
     return "#### $" + price

     


#Prediction Mode
if mode == "Prediction Mode":
    st.markdown("# Model Prediction")
    with st.expander("Important Info For Variables"):
         st.markdown("""
                     - Use only **integer** values for the **number of beds** (eg. 3)
                     - Use either **integer or .5** values for the **number of baths** (eg. 1.5)
                     - To choose **SW** as your area quadrant, leave all the other area quadrants as **0** 
                     - Do **not** select **more than one** area quadrant as having a value of **1**
                     """)
    start_model = st.sidebar.button("Initialize Model")
    model_choice = st.selectbox("Choose Regression Model", ['Basic OLS', 'OLS With Interaction Terms', 'Polynomial (Quadratic)'])
    col_sqr, col_bed, col_bath= st.columns(3, border= True) 
    col_se, col_nw, col_ne  = st.columns(3,border= True)
    col_predict = st.columns(1, border= True)
    with col_sqr: 
         sqr = st.number_input("**Choose Square Footage**")
    with col_bed:
         bed = st.number_input("**Choose Number of Beds**")
    with col_bath:
         bath = st.number_input("**Choose Number of Baths**")
    with col_se:
         s_e = st.number_input("**Choose Area Quadrant**   \n(1 = SE, 0 = Other Quadrant)")
    with col_nw:
         n_w= st.number_input("**Choose Area Quadrant**   \n(1 = NW, 0 = Other Quadrant)")  
    with col_ne:
         n_e = st.number_input("**Choose Area Quadrant**   \n(1 = NE, 0 = Other Quadrant)")
    with col_predict[0]:
         st.markdown("### Predicted House Price:")    
    if start_model:
         with col_predict[0]:
            if model_choice == 'Basic OLS':
                st.markdown(predict_linear(sqr, bed, bath, s_e, n_w, n_e))
            elif model_choice == 'OLS With Interaction Terms':
                 st.markdown(predict_interac(sqr, bed, bath, s_e, n_w, n_e))
            elif model_choice == 'Polynomial (Quadratic)':
                 st.markdown(predict_poly(sqr, bed, bath, s_e, n_w, n_e))
                 
               
         

     
           
