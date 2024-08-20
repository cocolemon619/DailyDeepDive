# 4.1は読んだ

# 4.2から
# 4.2.1
import numpy as np

def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 「2」を正解とする
t = [0,0,1,0,0,0,0,0,0,0]

# 例1：「2」の確率が最も高い場合(0.6)
y1 = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(sum_squared_error(np.array(y1),np.array(t)))

# 例2：「7」の確率が最も高い場合(0.6)
y2 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(sum_squared_error(np.array(y2),np.array(t)))

# 4.2.2
def cross_entropy_error(y, t):
    # 以下、deltaは、np.log0が-infとなり、それ以上の計算が出来なくなることを防ぐための処理
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

print(cross_entropy_error(np.array(y1), np.array(t)))
print(cross_entropy_error(np.array(y2), np.array(t)))