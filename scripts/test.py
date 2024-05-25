import numpy as np

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8], [9,10], [11,12]])
result_dot = np.dot(a, b)
print(result_dot)


a = np.array([[7,9,11],[8,10,12]])
b = np.array([[1,4], [2,5], [3,6]])
result_dot = np.dot(a, b)
# print(result_dot)
print(np.transpose(result_dot))


# data=np.arange(24).reshape(2,3,4)
# print(data)
# dataT=data.transpose((1,0,2))
# print(dataT)


# data=np.arange(48).reshape(2,3,4,2)
# print(data)
# print("----------------")
# dataT=data.transpose((0,2,1,3))
# print(dataT)