import pandas as pd
import numpy as np
rank = [1,1,1,1,1,2,2,2,2,2,3,3,3,4]
cd =   np.array([1,4,3,2,1,2,4,6,7,9,8,2,1,0])*-1

rng = np.random.default_rng(12345)
points = np.arange(0,14,1)

l = np.array([rank, cd, points]).T
import operator
s = sorted(l, key = operator.itemgetter(0,1))

print(np.array(l))
print("****")
print(np.array(s))
# # print(points.shape)
# df = pd.DataFrame({'rank' : rank, 'cd' : cd,'points' : points}).set_index('rank')

# print(df)

# df.sort_values(by = ['points', 'rank'], ascending = [False, True])
# print(df)