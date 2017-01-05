#!/usr/bin/env python
# -*- coding:utf-8 -*-
from perceptron import Perceptron

'''
定义激活函数f
'''
f = lambda x: x
class LinearUnit(Perceptron):
	def __init__(self, input_num):
		'''初始化线性单元，设置输入参数的个数'''
		Perceptron.__init__(self, input_num, f)

def get_training_dataset():

	#构建训练数据
	#输入向量列表:工作年限
	input_vecs = [[5], [3], [8], [1.4], [10.1]]
	#月薪
	labels = [5500,2300,7600,1800,11400]
	return input_vecs, labels

def train_linear_unit():
	'''
	使用数据训练线性单元
	'''
	#创建感知器，输入参数个数为1
	lu = LinearUnit(1)
	#训练迭代次数为10，学习速率为0.1
	input_vecs, labels =  get_training_dataset()
	
	lu.train(input_vecs, labels, 500, 0.01)
	
	#返回训练好的感知器
	return lu
	

	
if __name__ == '__main__':
	#训练线性单元
	linear_unit = train_linear_unit()
	#打印训练得到的权重
	print (linear_unit)
	
	#测试
	print ('work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
	print ('work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
	print ('work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
	print ('work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
		