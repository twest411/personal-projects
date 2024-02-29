from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

n = 100
xLowerLimit = -50
xUpperLimit = 50

x = np.random.rand(n,1)*(xUpperLimit - xLowerLimit) + xLowerLimit
y = 5*x + 7
y = y + np.random.rand(n,1)*(xUpperLimit - xLowerLimit)/2-(xUpperLimit - xLowerLimit)/4

model = LinearRegression().fit(x,y)

print('y = %.2f'%model.coef_[0],'* x + 2f'%model.intercept_)
MSE = sum((model.coef_[0]*x + model.intercept_ - y.reshape(-1,1))**2)/len(x)
print('MSE = %.4f'%MSE[0])

y_pred = model.predict(x)
plt.scatter(x,y, color='black')
plt.plot(x, y_pred, color='blue', linewidth=1)
plt.title('Linear Regression')
plt.xlabel('x-data')
plt.ylabel('y-data')
plt.show()