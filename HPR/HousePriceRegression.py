import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv("kc_house_data.csv")
data.head()
data.describe()

data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of bedrooms')
plt.xlabel('bedrooms')
plt.ylabel('count')
sns.despine

plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values ,y=data.long.values, size=10)
plt.xlabel('longtiude')
plt.ylabel('latitude')
plt.show()
sns.despine


plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")

#then the long isn't useful 
plt.scatter(data.price,data.long)
plt.title("Price vs Location of the area")

#isn't useful
plt.scatter(data.price,data.lat)
plt.title("Price vsLatitude")


plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


#isn't useful
plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")


train1 = data.drop(['id', 'price' , 'long' , 'lat'  , 'waterfront'],axis=1)
train1.head()


data.floors.value_counts().plot(kind='bar')
plt.scatter(data.floors,data.price)

plt.scatter(data.condition,data.price)

#seams a little bit useful
plt.scatter(data.zipcode,data.price)
plt.title("Which is the pricey location by zipcode?")


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
y=data['price']

conv_dates = [1 if values == 2014 else 0 for values in data.date ]

data['date']=conv_dates
train1=data.drop(['id', 'price' , 'long' , 'lat'  , 'waterfront' ,'floors'] , axis=1)


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(train1 , y , test_size = 0.10,random_state =2)

reg.fit(x_train,y_train)

reg.score(x_test,y_test)


