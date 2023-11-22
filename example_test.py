"""
可以解决任何类型问题，非线性规划、连续变量、整数变量、01变量均可（可以有等式和不等式约束条件）
"""
import numpy as np
from GA import GA

#设置目标函数
def f(X):
    return X[0]**2+X[1]**2+8

#设置不等式约束条件（默认都是要小于等于0）
def ineq_constraints(X):
    #约束1
    cons1=X[1]-X[0]**2
    return [cons1]

#设置等式约束（默认都是要小于等于0）
def eq_constraints(X):
    #约束2
    cons2=-X[0]-X[1]**2+2
    return [cons2] #组合成序列的形式

if __name__ == '__main__':
    varbound = np.array([[0,100], [0, 100]])  # 变量值域范围
    vartype = np.array([['real'],['real']])  # 变量类型
    #vartype = np.array([['real'], ['int'], ['int']])  # 变量类型可选，如果01变量也是选择int，变量值域限制为01即可

    #默认是最小化目标函数
    model = GA(function=f, #目标函数
               dimension=2, #决策变量数量
               variable_type=vartype, #变量类型（序列形式，real,int）
               variable_boundaries=varbound, #各变量值域
               eq_constraints=eq_constraints, #等式约束函数
               ineq_constraints=ineq_constraints, #不等式约束函数
               function_timeout=10, #延时（函数没响应）
               eq_cons_coefficient=0.001, #等式约束系数（值越小，等式约束越严格）
               max_num_iteration=300, #最大迭代次数
               population_size=1000, #个体数量（初始解的数量）
               penalty_factor=1, #惩罚因子（用于约束条件）,越大约束作用越大（要选择合适的值1比较合适）
               mutation_probability=0.1, #变异概率
               elit_ratio=0.01, #精英选择的比例（每轮都保留一定数量的较优目标函数的个体）
               crossover_probability=0.5, #交叉概率
               parents_portion=0.3, #父代比例(用于每代的交叉和变异)
               crossover_type='uniform', #交叉类型（”one_point“ ”two_point“ "uniform"）
               max_iteration_without_improv=None, #多少轮没更新后退出
               convergence_curve=True, #是否绘制算法迭代收敛曲线
               progress_bar=True, #进度条显示
               plot_path=None #保存收敛曲线svg图像的路径
               )
    model.run()
    #输出信息
    best_variable=model.best_variable #最优解
    best_function=model.best_function #最优目标函数
    report=model.report #每轮的最优目标函数
