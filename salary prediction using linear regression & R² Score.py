#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[10]:


sy = pd.read_csv("ML data.csv")


# In[11]:


print(sy)


# In[12]:


print(sy.dtypes)


# In[13]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
sy['Rename'] = le.fit_transform(sy['Name'])


# In[14]:


print(sy.dtypes)


# In[15]:


sy = sy.drop('Name', axis=1)


# In[16]:


print(sy.dtypes)


# In[17]:


sy['Rename'] = sy['Rename'].astype('int64')  #changing the data type from int32 to int64


# In[18]:


print(sy.dtypes)


# In[19]:


print(sy)


# In[20]:


x = sy[['Age']]             # Feature (can also try with ['Age', 'Rename'])
y = sy['Salary']            # Target


# In[21]:


print(x,y)


# In[22]:


# Step 2: Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[23]:


model = LinearRegression()    #linear regression
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[24]:


print(y_pred)


# In[25]:


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")


# In[32]:


plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary - Linear Regression')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


#R² Score (Coefficient of Determination)


# In[33]:


from sklearn.metrics import r2_score       #R² Score (Coefficient of Determination)
                                           #The R² score tells you how well the model explains the variance in the target
                                           #variable (Salary). It ranges from:
                                           #1.0 → Perfect fit
                                           #0.0 → Model explains none of the variance
                                           #< 0 → Worse than a horizontal line
print(r2_score(y_test, y_pred))


# In[35]:


r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")


# In[36]:


import matplotlib.pyplot as plt

residuals = y_test - y_pred

plt.scatter(y_pred, residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()


# In[ ]:




