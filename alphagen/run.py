import levy
import numpy as np

# 生成单个Levy随机数
levy_num = levy.levy(0, 1, 1)
print(levy_num)

# 生成一个Levy随机数数组
levy_array = levy.levy_flight(1000, 0, 1, 1)
print(levy_array)