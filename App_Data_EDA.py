import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def load_clean_data():
    data = pd.read_csv('../kc_house_data.csv')
    data.date = pd.to_datetime(data.date, infer_datetime_format=True)
    return data


data = load_clean_data()


def show_EDA_page():
    st.title('EDA of the KC House Sales Price Data')
    st.write("""In this page we present a brief EDA of the used data set, for more detailed EDA please refer
                to this [kernel](https://www.kaggle.com/hamzaboulahia/eda-kc), and the data set is available publicly 
                on [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction)""")

    st.write("Top 5 rows of the King County House Sales Price data set:")
    st.write(data.head())

    st.markdown(""" ## Features description:
**id :** The identification number of a house  
**date:** The date when a house was sold  
**price:** Price is prediction target  
**bedrooms:** Number of bedrooms  
**bathrooms:** Number of bathrooms  
**sqft_living:** Square footage of the home  
**sqft_lot:** Square footage of the lot  
**floors:** Total floors (levels) in house  
**waterfront:** House which has a view to a waterfront  
**view:** Number of views in the house  
**condition:** How good the house condition is overall  
**grade:** Overall grade given to the housing unit, based on King County grading system  
**sqft_above:** Square footage of house apart from basement  
**sqft_basement:** Square footage of the basement  
**yr_built:** Built Year  
**yr_renovated :** Year when house was renovated  
**zipcode:** Zip code  
**lat:** Latitude coordinate  
**long:** Longitude coordinate  
**sqft_living15:** Living room area in 2015(implies-- some renovations)  
**sqft_lot15:** LotSize area in 2015(implies-- some renovations)""")

    Years = list(pd.DatetimeIndex(data.date).year)
    Months = list(pd.DatetimeIndex(data.date).month)

    fig1 = plt.figure(figsize=(20, 6))
    grid = plt.GridSpec(2, 2, width_ratios=(1, 2), height_ratios=(1, 5), hspace=0.2, wspace=0.2)
    Left_ax = fig1.add_subplot(grid[:, 0])
    Right_top = fig1.add_subplot(grid[0, 1])
    Right_bot = fig1.add_subplot(grid[1, 1], xticklabels=['Jan', 'Feb', 'Mar', 'May', 'Avr', 'Jun', 'Jul', 'Aou',
                                                         'Sep', 'Oct', 'Nov', 'Dec'])

    sb.countplot(x=Years, palette='mako', ax=Left_ax)
    Left_ax.set_title('House sales count by Year', fontdict={'fontsize': 15})
    sb.countplot(x=Months, palette='mako', ax=Right_bot)
    sb.boxplot(x=Months, ax=Right_top)
    Right_top.set_title('House sales count by Month', fontdict={'fontsize': 15})

    st.write("""## Univariate Data Exploration:""")
    st.write("Sales count by Year & Month")
    st.pyplot(fig1)

    st.write("House Price distribution")
    fig2 = plt.figure(figsize=(20, 8))
    plt.subplot(121)
    sb.histplot(data=data.price, bins=140)
    plt.title('Distribution of the house prices', fontdict={'fontsize': 15})

    plt.subplot(122)
    sb.boxplot(x=data.price)
    plt.title('Boxplot of the house prices', fontdict={'fontsize': 15})
    st.pyplot(fig2)

    st.write("Bedrooms & Bathrooms distribution")
    fig3 = plt.figure(figsize=(20, 6))
    plt.subplot(121)
    sb.countplot(x=data.bedrooms, palette='mako')
    plt.title('Number of bedrooms distribution', fontdict={'fontsize': 15})
    plt.subplot(122)
    sb.countplot(y=data.bathrooms, palette='mako')
    plt.title('Number of bathrooms distribution', fontdict={'fontsize': 15})
    st.pyplot(fig3)

    st.write("House area distribution")
    fig4 = plt.figure(figsize=(20, 15))
    sb.histplot(x=data.sqft_living, kde=True, bins=110)
    sb.histplot(x=data.sqft_living15, kde=True, bins=110, color='red')
    plt.legend(['sqft_living', 'sqft_living15'])
    plt.title('Living area distribution', fontdict={'fontsize': 15})
    st.pyplot(fig4)

    st.write("Classes representation for some categorical features")
    fig5 = plt.figure(figsize=(20, 20))
    plt.subplot(321)
    sb.countplot(x=data.floors, palette='mako')
    plt.title('Distribution of houses with respect to floor count', fontdict={'fontsize': 15})
    plt.subplot(322)
    sb.countplot(x=data.waterfront, palette='mako')
    plt.title('Number of houses with/without a water front', fontdict={'fontsize': 15})
    plt.subplot(323)
    sb.countplot(x=data.view, palette='mako')
    plt.title('Distribution of the views count', fontdict={'fontsize': 15})
    plt.subplot(324)
    sb.countplot(x=data.condition, palette='mako')
    plt.title('Houses condition distribution', fontdict={'fontsize': 15});
    st.pyplot(fig5)

    st.write("""## Multivariate Data Exploration:""")
    st.write("Feature correlation heatmap")
    fig6 = plt.figure(figsize=(18, 13))
    plt.title('Heatmap correlation of the most important features', fontsize=18)
    sb.heatmap(data=data.iloc[:, 1:].corr(), annot=True)
    st.pyplot(fig6)

    st.write("Price Vs Categorical variables")
    fig7 = plt.figure(figsize=(20, 20))
    plt.subplot(421)
    sb.barplot(x=data.bedrooms, y=data.price, palette='mako')
    plt.subplot(422)
    sb.barplot(x=data.waterfront, y=data.price, palette='mako')
    plt.subplot(423)
    sb.barplot(x=data.grade, y=data.price, palette='mako')
    plt.subplot(424)
    sb.barplot(x=data.floors, y=data.price, palette='mako')
    plt.subplot(425)
    sb.barplot(x=data.condition, y=data.price, palette='mako')
    plt.subplot(426)
    sb.barplot(x=data.view, y=data.price, palette='mako')
    plt.subplot(414)
    sb.barplot(x=data.bathrooms, y=data.price, palette='mako')
    st.pyplot(fig7)

    st.subheader("You can predict the price of a house using a Linear Regression model on the Predict page")

    return None
