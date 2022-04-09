import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.metrics import r2_score
while(True):
    df = pd.read_csv(r"C:\Users\persian\Downloads\FixData.csv")
    df['Area'] = df['Area'].astype(int) 
    df['Room'] = df['Room'].astype(int)             #Make Data Numberable
    df['Price(USD)'] = df['Price(USD)'].astype(int)
    df['Price'] = df['Price'].astype(int)
    le = preprocessing.LabelEncoder()
    le.fit(df["Address"])
    adres = le.transform(df["Address"])
    df["Address"]=adres
    msk = np.random.rand(len(df))<0.8
    train = df[msk]
    test = df[~msk]

    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[["Area" , "Room" ,"Parking","Warehouse","Elevator","Address"]])
    train_y = np.asanyarray(train[["Price(USD)"]])
    regr.fit(train_x , train_y)

    test_x = np.asanyarray(test[["Area" , "Room","Parking","Warehouse","Elevator","Address"]])
    test_y = np.asanyarray(test[["Price(USD)"]])
    test_y_ = regr.predict(test_x)
    if r2_score(test_y , test_y_) > 0.67 :
        print("r2_score is :" , r2_score(test_y, test_y_))
        print("Coefitions :" , regr.coef_)
        print("Intercept :" , regr.intercept_)
        break

Area = int(input("inter Area Of Home :"))
Room = int(input("inter Count of Room :"))
Parking =input("Parking? yes(1) or (0)")
Elevator = input("Elevator?")
Warehouse = input("Warehouse?")
a = input("Enter The Address :")
Address = le.transform([a])[0]
data = np.asanyarray([[Area , Room ,Parking ,Elevator,Warehouse,Address]])

y=regr.predict(data)
print("------------------------------------------------------------")
print("The Price Of Your Home With " ,round(r2_score(test_y, test_y_)*100) , "%" , "Accuracy is :" , round(y[0][0]) , "$")
print("------------------------------------------------------------")





#print(le.inverse_transform([156]))