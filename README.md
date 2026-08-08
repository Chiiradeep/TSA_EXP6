# Ex.No: 6               HOLT WINTERS METHOD
### Date: 08/08/2026



### AIM:

To implement Holt Winters Method

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
~~~py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset,perform data exploration
data = pd.read_csv('/content/air traffic.csv')

# Convert 'Year' and 'Month' columns to a datetime index
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str) + '-01')
data = data.set_index('Date')

# Convert 'Pax' to numeric by removing commas
data['Pax'] = data['Pax'].str.replace(',', '').astype(float)

data.head()

# Resample and plot data
data_monthly = data.resample('MS').sum(numeric_only=True)   #Month start
data_monthly.head()
data_monthly['Pax'].plot() # Plotting 'Pax' column specifically as other columns might not be numeric after sum for resampling

# Scale the data and check for seasonality
scaler = MinMaxScaler()
# Ensure 'Pax' column is selected for scaling if data_monthly contains other columns
scaled_data = pd.Series(scaler.fit_transform(data_monthly[['Pax']].values.reshape(-1, 1)).flatten()) # The data seems to have additive trend and multiplicative seasonality
scaled_data.index = data_monthly.index # Assign back the datetime index
scaled_data.plot()


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly['Pax'], model="additive") # Decompose 'Pax' column
decomposition.plot()
plt.show()

# Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data

s_data=scaled_data+1 # Adding 1 as scaling might result in 0 which can cause issues with multiplicative models
train_data = s_data[:int(len(s_data) * 0.8)]
test_data = s_data[int(len(s_data) * 0.8):]

# For ExponentialSmoothing, ensure the index is passed for proper time series handling
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit() # Assuming monthly data, seasonal_periods=12

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')


rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
mae = mean_absolute_error(test_data, test_predictions_add)
std_dev = np.sqrt(s_data.var())

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"Standard Deviation: {std_dev}")

# Create the final model and predict future data and plot it

final_model = ExponentialSmoothing(data_monthly['Pax'], trend='add', seasonal='mul', seasonal_periods=12).fit()

final_predictions = final_model.forecast(steps=int(len(data_monthly)/4))

ax=data_monthly['Pax'].plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Months')
ax.set_ylabel('Number of monthly passengers') # Swapped labels as per standard practice, X for time, Y for values
ax.set_title('Prediction')
~~~


### OUTPUT:

## TEST_PREDICTION

<img width="534" height="448" alt="fd0f74cd-0b74-42e5-bd03-f754cd3792bc" src="https://github.com/user-attachments/assets/8b2eec86-bab8-417b-a733-e34480e5a6e3" />


<img width="630" height="470" alt="c8bcdc9e-c892-4b1b-8f27-9572fe34fedb" src="https://github.com/user-attachments/assets/1733d270-5a38-444e-b794-0ceae0917902" />




## FINAL_PREDICTION

<img width="1116" height="798" alt="Screenshot 2026-08-08 133418" src="https://github.com/user-attachments/assets/9d3c708e-ab51-4775-93b2-42815005105f" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
