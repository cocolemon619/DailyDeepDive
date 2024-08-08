# 3.2から
import numpy as np

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_list(x): # 引数xはlist形式
    y = x > 0
    # return y.astype(int) がAttributeErrorで実行できなかったため、以下で実行
    return y.astype(int) # list形式で与えると大丈夫だったけど、list形式で与えないとだめだった。

print(step_function(3.4))
print(step_function_list(np.array([3,4,3.3])))
