# Cài đặt giải thuật Gradient Descent tìm min của hàm 1 biến:
from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

#ham so f(x) = x^2 + 5*sin(x)  f'(x) = 2*x + 5*cos(x)

def grad(x):   #tinh dao ham
    return 2*x+ 5*np.cos(x)

def cost(x):   #hàm tính giá trị của hàm số (để kiểm tra xem giá trị của hàm số có giảm theo mỗi vòng lặp không)
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):  #thuật toán Gradient Descent, eta: tốc độ học, x0: điểm bắt đầu
    x = [x0]   # x là 1 list có phần tu x0 bên trong
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])  #x[-1] phần tử cuối cùng của list x
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)   #thêm x_new vào x
    return (x, it)  #trả về list x là số lần lặp để được kết quả

(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
#nhận xét: chọn x0 = -5 hội tụ nhanh hơn (11 lần lặp) x0 = 5 (29 lần lặp)

(x1, it1) = myGD1(.1, -3)
(x2, it2) = myGD1(.1, 3)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
# nhận xét????

(x1, it1) = myGD1(.5, -5)
(x2, it2) = myGD1(.5, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
#nhận xét????

(x1, it1) = myGD1(.01, -5)
(x2, it2) = myGD1(.01, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
#nhận xét????

#Nhận xét toàn bài: Tốc độ hội tụ của GD không chỉ phụ thuộc vào x0 mà còn phụ thuộc vào eta (learning rate)
# -> lựa chọn eta là rất quan trọng (cần thí nghiệm để lựa chọn), thậm chí có thể chọn eta khác nhau ở mỗi vòng lặp
