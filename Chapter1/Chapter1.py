# # 1.4
# print("I'm hungry!")

# # man.py の実行
# from man import Man

# m = Man("David")
# m.hello()
# m.goodbye()

# # 1.5
# import numpy as np
# x = np.array([1.0 , 2.0 , 3.0])
# print(x)
# print(type(x))

# x = np.array([1.0 , 2.0 , 3.0])
# y = np.array([2.0 , 4.0 , 6.0])
# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)

# x = np.array([1.0, 2.0, 3.0])
# print(x/2.0)

# A = np.array([[1,2],[3,4]])
# print(A)
# print(A.shape)
# print(A.dtype)

# B = np.array([[3,0],[0,6]])
# print(A + B)
# print(A * B)

# print(A * 10)

# A = np.array([[1,2],[3,4]])
# B = np.array([10,20])
# print(A * B)

# X = np.array([[51,55],[14,19],[0,4]])
# print(X)
# print(X[0])
# print(X[0][1])

# for row in X:
#     print(row)

# X = X.flatten()
# print(X)
# print(X[np.array([0,2,4])])

# print(X > 15)
# print(X[X>15])

# 1.6
# import numpy as np
# import matplotlib.pyplot as plt

# x = np.arange(0,6,0.1)
# y = np.sin(x)

# plt.plot(x, y)
# plt.show()

# x = np.arange(0,6,0.1)
# y1 = np.sin(x)
# y2 = np.cos(x)

# plt.plot(x,y1,label="sin")
# plt.plot(x,y2,linestyle="--",label="cos")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title('sin & cos')
# plt.legend()
# plt.savefig("./Chapter1/sin & cos.jpg") # 参考書にはないけど、自身で追加
# plt.show()

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("./Chapter1/sin & cos.jpg")
plt.imshow(img)

plt.show()