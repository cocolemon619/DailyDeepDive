# 3.2から
import numpy as np

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function2(x):
    y = x > 0
    # return y.astype(int) がAttributeErrorで実行できなかったため、以下で実行
    return int(y)

print(step_function(3.4))
print(step_function2(3.4))

# print(step_function(np.array([3,4,3.3]))) # 
print(step_function2(np.array([3,4,3.3])))
