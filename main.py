import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle as pk



data = pd.read_csv('score.csv')

print(data.iloc[0])
# print((max(data.Hours)))
# plt.scatter(data.Hours, data.Scores)
# plt.show()

costgraph = []

def gradient_descent(mNow, bNow, learningRate, data, iter):
    mGradient, bGradient = 0, 0
    error = 0
    n = len(data)
    
    for i in range(n):
        x = data.iloc[i].Hours
        y = data.iloc[i].Scores
        # mes = sum(map(lambda dat: (y - (mNow * dat.Scores + bNow)) ** 2, data)) / n
        
        if iter % 50 == 0:
            error += (y - (mNow * x + bNow)) ** 2
        # print(mes)
        
        mGradient += -(2/n) * x * (y- (mNow * x + bNow))
        bGradient += -(2/n) * (y- (mNow * x + bNow))
        
    m = mNow - mGradient * learningRate
    b = bNow - bGradient * learningRate
    if iter % 50 == 0:
        mes = error / n
    
        costgraph.append(mes)
    
    return m, b

m, b = 0, 0
learningRate = 0.0001
epochs = 1000
xMin = math.floor(min(data.Hours))
xMax = math.ceil(max(data.Hours))

for i in range(epochs):
    if i % 100 == 0:
        print(f"epochs: {i}") 
        print(f"m = {m}, b = {b}")
    plt.show()
    m, b = gradient_descent(m, b, learningRate, data, i)
    

    
    
plt.scatter(data.Hours, data.Scores, color="black")
plt.plot(list(range(xMin, xMax)), [m * x + b for x in range(xMin, xMax)])
plt.show()

plt.plot(list(range(0, len(costgraph))), costgraph)
plt.show()

