# 3.2から
import numpy as np

# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# def step_function_list(x): # 引数xはlist形式
#     y = x > 0
#     # return y.astype(int) がAttributeErrorで実行できなかったため、以下で実行
#     return y.astype(int) # list形式で与えると大丈夫だったけど、list形式で与えないとだめだった。

# print(step_function(3.4))
# print(step_function_list(np.array([3,4,3.3])))

# # 3.2.3から
# import matplotlib.pylab as plt

# def step_function(x):
#     return np.array(x>0,dtype = int)

# x = np.arange(-5.0,5.0,0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1)
# plt.savefig("Chapter3/step_function.jpg") # 自身で追加
# plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# x = np.array([-1.0, 1.0, 2.0])
# print(sigmoid(x))

# t = np.array([1.0, 2.0, 3.0])
# print(1.0 + t)
# print(1.0 / t)

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.savefig("Chapter3/sigmoid.jpg")
# plt.show()

# x = np.arange(-5.0,5.0,0.1)
# y1 = step_function(x)
# y2 = sigmoid(x)

# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.ylim(-0.1, 1.1)
# plt.savefig("Chapter3/step_function & sigmoid.jpg")
# plt.show()

# def relu(x):
#     return np.maximum(0,x)

# # 3.3.1
# A = np.array([1,2,3,4])
# print(A)
# print(np.ndim(A))
# print(A.shape)
# print(A.shape[0])

# B = np.array([[1,2],[3,4],[5,6]])
# print(B)
# print(np.ndim(B))
# print(B.shape)

# # 3.3.2
# A = np.array([[1,2],[3,4]])
# print(A.shape)
# B = np.array([[5,6],[7,8]])
# print(B.shape)
# print(np.dot(A,B))

# A = np.array([[1,2,3],[4,5,6]])
# print(A.shape)
# B = np.array([[1,2],[3,4],[5,6]])
# print(B.shape)
# print(np.dot(A,B))

# C = np.array([[1,2],[3,4]])
# print(C.shape)
# print(A.shape)
# # print(np.dot(A,C)) # 実行結果、行列Aの1次元目と行列Cの0次元目の次元の要素数が一致していないため、エラー
# print(np.dot(C,A))

# A = np.array([[1,2],[3,4],[5,6]])
# print(A.shape)
# B = np.array([7,8])
# print(B.shape)
# print(np.dot(A,B))

# # 3.3.3
# X = np.array([1,2])
# print(X.shape)
# W = np.array([[1,3,5],[2,4,6]])
# print(W)
# print(W.shape)
# Y = np.dot(X, W)
# print(Y)

# # 3.4
# X = np.array([1.0, 0.5])
# W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
# B1 = np.array([0.1,0.2,0.3])
# print(W1.shape)
# print(X.shape)
# print(B1.shape)

# A1 = np.dot(X, W1) + B1
# print(A1)

# Z1 = sigmoid(A1)
# print(A1)
# print(Z1)

# W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
# B2 = np.array([0.1, 0.2])

# print(Z1.shape)
# print(W2.shape)
# print(B2.shape)

# A2 = np.dot(Z1, W2) + B2
# Z2 = sigmoid(A2)

def identity_fuction(x):
    return x

# W3 = np.array([[0.1,0.3],[0.2,0.4]])
# B3 = np.array([0.1,0.2])

# A3 = np.dot(Z2, W3) + B3
# Y = identity_fuction(A3)
# print(Y)

# # 3.4.3
# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
#     network['b1'] = np.array([0.1,0.2,0.3])
#     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     network['b2'] = np.array([0.1, 0.2])
#     network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
#     network['b3'] = np.array([0.1,0.2])
    
#     return network

# def forward(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = identity_fuction(a3)
    
#     return y

# network = init_network()
# print(network)
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y)

# # 3.5.1
# a = np.array([0.3, 2.9, 4.0])
# exp_a = np.exp(a)
# print(exp_a)
# sum_exp_a = np.sum(exp_a)
# print(sum_exp_a)
# y = exp_a / sum_exp_a
# print(y)

# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
    
#     return y

# # 3.5.2
# a = np.array([1010, 1000, 990])
# print(np.exp(a) / np.sum(np.exp(a))) # >> array([nan, nan, nan])
# c = np.max(a) # 1010
# print(a - c) # >> array([0, -10, -20])
# print(np.exp(a - c) / np.sum(np.exp(a - c)))

# # 3.5.3
# # オーバーフロー対策をしたソフトマックス関数
# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c) # オーバーフロー対策
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
    
#     return y

# a = np.array([0.3, 2.9, 4.0])
# y = softmax(a)
# print(y)
# print(np.sum(y))

