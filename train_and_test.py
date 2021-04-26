import numpy as np
import pandas as pd
from nn import *

#For IRIS Dataset
data = pd.read_csv("Data/Iris.data", header = 0)
print(data.describe())

#normalizing the data
df_norm = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(df_norm.sample(n=5))


target = data[['class']].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])
print(target.sample(n=5))

df = pd.concat([df_norm, target], axis=1)
print(df.sample(n=5))

train_test_per = 90/100.0
df['train'] = np.random.rand(len(df)) < train_test_per
print(df.sample(n=5))


train = df[df.train == 1]
train = train.drop('train', axis=1).sample(frac=1)
print(train.sample(n=5))



test = df[df.train == 0]
test = test.drop('train', axis=1)
print(test.sample(n=5))


x_train = train.values[:,:4]
print(x_train[:5])

targets = [[1,0,0],[0,1,0],[0,0,1]]
y_train = np.array([targets[int(x)] for x in train.values[:,4:5]])
print(y_train[:5])

x_test = test.values[:,:4]
print(x_test[:5])

targets = [[1,0,0],[0,1,0],[0,0,1]]
y_test = np.array([targets[int(x)] for x in test.values[:,4:5]])
print(y_test[:5])
nn = NeuralNetwork(learning_rate=0.1, debug=False)
nn.add_layer(n_inputs=len(x_train[0]), n_neurons=3)

nn.train(dataset=x_train,targets=y_train, n_iterations=100, print_error_report=True)

nn.test(dataset=x_test,targets=y_test)


#For breast cancer wisconsin dataset
df_1 = pd.read_csv("Data/breast_cancer_wisconsin.csv", header = 0)

df_1.drop('id', axis=1, inplace=True)
df_1.drop(['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'], axis=1, inplace=True)

df_1.drop(['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se' ,
         'symmetry_se', 'fractal_dimension_se'], axis=1, inplace=True)

df_1.drop(['perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean'], axis=1, inplace=True)


#normalizing the data
df_1_norm = df_1[['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(df_1_norm.sample(n=5))

target_1 = df_1[['diagnosis']].replace(['M','B'],[0,1])
print(target_1.sample(n=5))

df_1 = pd.concat([df_1_norm, target_1], axis=1)
print(df_1.sample(n=5))

train_test_per_1 = 90/100.0
df_1['train'] = np.random.rand(len(df_1)) < train_test_per_1
print(df_1.sample(n=5))


train_1 = df_1[df_1.train == 1]
train_1 = train_1.drop('train', axis=1).sample(frac=1)
print(train_1.sample(n=5))



test_1 = df_1[df_1.train == 0]
test_1 = test_1.drop('train', axis=1)
print(test_1.sample(n=5))


x_train_1 = train_1.values[:,:4]
print(x_train_1[:5])

targets_1 = [[1,0],[0,1]]
y_train_1 = np.array([targets_1[int(x)] for x in train_1.values[:,4:5]])
print(y_train_1[:5])

x_test_1 = test_1.values[:,:4]
print(x_test_1[:5])

targets_1 = [[1,0],[0,1]]
y_test_1 = np.array([targets_1[int(x)] for x in test_1.values[:,4:5]])
print(y_test_1[:5])

nn_1 = NeuralNetwork(learning_rate=0.1, debug=False)
nn_1.add_layer(n_inputs=len(x_train_1[0]), n_neurons=2)

nn_1.train(dataset=x_train_1,targets=y_train_1, n_iterations=100, print_error_report=True)
nn_1.test(dataset=x_test_1,targets=y_test_1)




