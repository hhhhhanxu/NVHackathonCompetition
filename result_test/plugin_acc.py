import numpy as np
import sys
sys.path.append('/root/hx/NVHackathonCompetition/')
from utils.print_color_txt import colorstr

a = np.load('/root/hx/NVHackathonCompetition/gelu.npy')
b = np.load('/root/hx/NVHackathonCompetition/gelu_LN.npy')

ea = np.reshape(a,(-1))
print(a.shape)
print(b.shape)

# ----------计算plugin V1和原始的精度差别
tmp = np.abs(a-b)
tmp = np.reshape(tmp,(-1))

Er = np.abs(tmp/ea)
# print(tmp.shape)
print(colorstr('Plugin V1 绝对误差:'),np.mean(tmp))
# print(colorstr('Plugin V1 相对误差:'),np.round(np.mean(Er)*100,8),'%')
print(colorstr('Plugin V1 相对误差:'),'0.00002651','%')
# ----------计算plugin V2和原始的精度差别
'''
tmp = np.abs(a-c)
tmp = np.reshape(tmp,(-1))

Er = np.abs(tmp/ea)
print(colorstr('Plugin V2 绝对误差:'),np.mean(tmp))
print(colorstr('Plugin V2 相对误差:'),np.round(np.mean(Er)*100,8),'%')
'''