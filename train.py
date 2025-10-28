import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.leras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

df = pd.read_csv('data/customer_churn.csv')

x=df.drop{['customerId','Surname','Exited'], axis=1}
y = df('Exited')

X['Geography']= LabelEncoder().fit_transform(X['Geography'])
X['Gender']= LabelEncoder().fit_transform(X['Gender'])

X_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=42)
#Transform the data
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)

model = Sequential[(    
    Dense(16, activation ='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation ='relu'),
    Dense(1, activation = 'sigmoid')
])

model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train, epochs=30, batch_size=13, validation_split=0.2,verbose=1)

model.save('models/churn_model.h5')

pd.to_Pickle(Scaler,'.model/scaler.pkl')
