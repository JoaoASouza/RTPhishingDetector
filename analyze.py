import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file1 = pd.read_csv('out.csv')
file2 = pd.read_csv('out2.csv')

data1 = np.array(file1[['answer_count', 'ttl_avg']].values)
data2 = np.array(file2[['answer_count', 'ttl_avg']].values)

plt.scatter(data1[:, 0], data1[:, 1], c='b', s=2, label='legitimos')
plt.scatter(data2[:, 0], data2[:, 1], c='r', s=2, label='maliciosos')
plt.xlabel("quantidade")
plt.ylabel("TTL médio")
plt.legend()
plt.show()

print(data1)
print(data2)