#%%
import pandas as pd

df = pd.read_csv('social_media_mental_health.csv')
#%%
df.head()
# %% попытка в линейную регресию
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df[['Daily_Screen_Time_Hours','Sleep_Duration_Hours' ]]
Y = df['PHQ_9_Score']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

# %%

model= LinearRegression()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse= mean_squared_error(Y_test, y_pred)
r2= r2_score(Y_test, y_pred)


coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nВлияние признаков:\n", coefficients)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# %%
