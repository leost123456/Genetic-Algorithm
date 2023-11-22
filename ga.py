"""
本项目基于rmsolgi的geneticalgorithm项目进行改进，实现各种约束条件下的目标优化(加入罚函数)，原始项目信息请见：https://github.com/rmsolgi/geneticalgorithm
Copyright 2020 Ryan (Mohammad) Solgi

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import sys
import time

from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams #导入包

config = {"font.family":'Times New Roman'}  # 设置字体类型
rcParams.update(config)    #进行更新配置

class GA():
    def __init__(self, function, dimension,
                 variable_boundaries=None,
                 variable_type=None,
                 eq_constraints=None,
                 ineq_constraints=None,
                 function_timeout=10,
                 eq_cons_coefficient=0.001,
                 max_num_iteration=None,
                 population_size=100,
                 penalty_factor=1,
                 mutation_probability=0.1,
                 elit_ratio=0.01,
                 crossover_probability=0.5,
                 parents_portion=0.3,
                 crossover_type='uniform',
                 max_iteration_without_improv=None,
                 convergence_curve=True,
                 progress_bar=True,
                 plot_path=None,**kwargs):

        self.plot_path = plot_path  # 存储图像的路径

        self.__name__ = GA

        #目标函数
        assert (callable(function)), "function must be callable"
        self.f = function

        #决策变量数量
        self.dim = int(dimension)

        #约束条件
        self.eq_constraints = eq_constraints  # 注意约束为序列形式,里面是约束函数(等式约束)
        self.ineq_constraints = ineq_constraints #不等式约束

        #等式约束系数（用于限制等式约束）越小越严格
        self.eq_cons_coefficient=eq_cons_coefficient

        # 输入变量的类型（连续变量、整数变量）
        assert variable_type is not None,'Input variable type cannot be empty'
        assert (type(variable_type).__module__ == 'numpy'),"\n variable_type must be numpy array"
        assert (len(variable_type) == self.dim), "\n variable_type must have a length equal dimension."
        for i in variable_type:  # 确保都要为连续变量或者整数变量，如果是01变量就更改其变量的范围即可
            assert (i == 'real' or i == 'int'), \
                "\n variable_type_mixed is either 'int' or 'real' " + \
                "ex:['int','real','real']" + \
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"
        self.var_type = variable_type

        # 设置变量的范围 input variables' boundaries
        assert (type(variable_boundaries).__module__ == 'numpy'), "\n variable_boundaries must be numpy array"
        assert (len(variable_boundaries) == self.dim), "\n variable_boundaries must have a length equal dimension"
        for i in variable_boundaries:
            assert (len(i) == 2), "\n boundary for each variable must be a tuple of length two."
            assert (i[0] <= i[1]), "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
        self.var_bound = variable_boundaries

        # Timeout 超时
        self.funtimeout = float(function_timeout)

        # 收敛曲线 convergence_curve
        if convergence_curve == True:
            self.convergence_curve = True
        else:
            self.convergence_curve = False

        # 进度条 progress_bar
        if progress_bar == True:
            self.progress_bar = True
        else:
            self.progress_bar = False

        # 下面是输入的超参数部分
        self.penalty_factor = penalty_factor #惩罚因子
        self.pop_s = int(population_size) #个体数量

        assert (parents_portion <= 1 and parents_portion >= 0),"parents_portion must be in range [0,1]"
        self.par_s = int(parents_portion * self.pop_s)  #下一代中保留的父代数量
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = mutation_probability # 变异概率
        assert (self.prob_mut <= 1 and self.prob_mut >= 0), "mutation_probability must be in range [0,1]"

        self.prob_cross = crossover_probability  # 交叉概率
        assert (self.prob_cross <= 1 and self.prob_cross >= 0), "cross_probability must be in range [0,1]"

        assert (elit_ratio <= 1 and elit_ratio >= 0), "elit_ratio must be in range [0,1]"
        trl = self.pop_s * elit_ratio
        if trl < 1 and elit_ratio > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)  # 精英选择数量

        assert (self.par_s >= self.num_elit), "\n number of parents must be greater than number of elits"

        if max_num_iteration == None:  # 控制最大迭代轮数
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == 'int':
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * self.dim * (100 / self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * 50 * (100 / self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(max_num_iteration)

        self.c_type = crossover_type  # 交叉的方式（单点交叉、双点交叉和均匀交叉）默认为均匀交叉
        assert (self.c_type == 'uniform' or self.c_type == 'one_point' or self.c_type == 'two_point'), \
            "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

        self.stop_mniwi = False
        if max_iteration_without_improv == None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(max_iteration_without_improv)  # 最大没有更新最优解的轮数

    # 主要的算法主体部分
    def run(self):
        # 初始化种群Initial Population
        self.integers = np.where(self.var_type == 'int')  # 存储整数变量的序号位置
        self.reals = np.where(self.var_type == 'real')  # 存储连续变量的序号位置

        pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)  # 产生的种群，形状为（个体数量，变量数量+1）每个个体的最后一个位置用于存储适应度
        solo = np.zeros(self.dim + 1)  # 存储单个个体的一个解和其对应的适应度，最后一个位置用于保存适应度
        var = np.zeros(self.dim)  # 单个个体的决策变量

        # 下面进行随机生成各个变量在值域范围内的值
        for p in range(0, self.pop_s):  # 每个个体

            for i in self.integers[0]:  # 每个整数变量的序号
                var[i] = np.random.randint(self.var_bound[i][0], \
                                           self.var_bound[i][1] + 1)
                solo[i] = var[i].copy()
            for i in self.reals[0]:  # 每个连续变量的序号
                var[i] = self.var_bound[i][0] + np.random.random() * \
                         (self.var_bound[i][1] - self.var_bound[i][0])
                solo[i] = var[i].copy()

            obj = self.sim(var)  # 计算适应度（目标函数值）
            solo[self.dim] = obj
            pop[p] = solo.copy()  # 存储

        # Report
        self.report = [] #每轮的最优解
        self.best_variable = var.copy()  # 最优解
        self.best_function = obj  # 最优目标函数
        with tqdm(total=self.iterate, desc="Processing", unit="iteration") as pbar: #设置进度条
            t = 1 #统计迭代轮数
            counter = 0 #统计未更新的轮数
            while t <= self.iterate:  # 每轮更新迭代
                # Sort
                pop = pop[pop[:, self.dim].argsort()]  # 根据最后一个目标函数进行升序排列

                # 更新目前目标函数值最小的解向量和对应目标函数值
                if self.eq_constraints is None and self.ineq_constraints is None:  # 无约束条件
                    if pop[0, self.dim] < self.best_function:
                        counter = 0
                        self.best_function = pop[0, self.dim].copy()
                        self.best_variable = pop[0, : self.dim].copy()
                    else:
                        counter += 1

                else:  # 有约束条件
                    for i in range(len(pop)):  # 每个个体
                        if self.penalty(pop[i, :self.dim]) <= 0:  # 判断是否可行解
                            if pop[i, self.dim] < self.best_function:  # 最优解判断
                                counter = 0
                                self.best_function = pop[i, self.dim].copy()
                                self.best_variable = pop[i, : self.dim].copy()
                                break
                            else:
                                counter += 1
                                break
                    else:
                        counter += 1

                # Report
                # 记录每轮迭代最优的目标函数值（每次都是历史最优）
                self.report.append(pop[0, self.dim])

                # 标准化下目标函数 Normalizing objective function
                normobj = np.zeros(self.pop_s)  # 用于存储修正后的目标函数序列

                minobj = pop[0, self.dim]  # 最小目标函数
                # 确保所有目标函数大于0
                if minobj < 0:
                    normobj = pop[:, self.dim] + abs(minobj)

                else:
                    normobj = pop[:, self.dim].copy()

                maxnorm = np.amax(normobj)  # 最大目标函数值
                normobj = maxnorm - normobj + 1  # 最优的目标函数现在变最大（>0）

                # 进行计算概率（Calculate probability)
                sum_normobj = np.sum(normobj)
                prob = np.zeros(self.pop_s)
                prob = normobj / sum_normobj  # 个体概率（适应度越大，则概率越大）
                cumprob = np.cumsum(prob)  # 累计概率

                # 选择父代（Select parents）
                par = np.array([np.zeros(self.dim + 1)] * self.par_s)  # 形状为（dim+1,设置留下父母数量）

                par[:self.num_elit] = pop[:self.num_elit].copy()  # 进行精英选择（将最好的几个个体直接作为父代保留）
                for k in range(self.num_elit, self.par_s):  # 其余父代通过概率随机筛选（轮盘赌法）
                    index = np.searchsorted(cumprob, np.random.random())  # 找到第一个大于或等于随机生成的概率值的位置。返回的索引表示该位置。
                    par[k] = pop[index].copy()

                ef_par_list = np.array([False] * self.par_s)  # 判断是否用于交叉的序列
                par_count = 0  # 用于交叉、变异操作的父代总数
                while par_count == 0:
                    for k in range(0, self.par_s):
                        if np.random.random() <= self.prob_cross:
                            ef_par_list[k] = True
                            par_count += 1

                ef_par = par[ef_par_list].copy()  # 取出筛选出的用于后续交叉、变异的父代

                # 产生新一代(New generation)
                pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)

                # 将事先已经通过精英筛选和轮盘赌法后的优秀父代保留
                for k in range(0, self.par_s):
                    pop[k] = par[k].copy()

                # 利用筛选的父代进行交叉、变异操作并和选择的优秀父代共同形成新一代
                for k in range(self.par_s, self.pop_s, 2):  # 步长为2，每轮都会生成2个子代
                    r1 = np.random.randint(0, par_count)  # 随机选择一个父代序号1
                    r2 = np.random.randint(0, par_count)  # 随机选择一个父代序号2
                    pvar1 = ef_par[r1, : self.dim].copy()  # 父代解1
                    pvar2 = ef_par[r2, : self.dim].copy()  # 父代解2

                    ch = self.cross(pvar1, pvar2, self.c_type)  # 进行交叉操作，返回两个子代解
                    ch1 = ch[0].copy()  # 产生的第一个子代解
                    ch2 = ch[1].copy()  # 产生的第二个子代解

                    ch1 = self.mut(ch1)  # 对第一个子代进行随机变异
                    ch2 = self.mutmidle(ch2, pvar1, pvar2)  # 对第二个子代进行在两个父代之间的变异
                    # 进行存储子代解与其适应度
                    # 子代1
                    solo[: self.dim] = ch1.copy()
                    obj = self.sim(ch1)
                    solo[self.dim] = obj
                    pop[k] = solo.copy()
                    # 子代2
                    solo[: self.dim] = ch2.copy()
                    obj = self.sim(ch2)
                    solo[self.dim] = obj
                    pop[k + 1] = solo.copy()
                pbar.update(1)

                t += 1  # 迭代轮数+1
                if counter > self.mniwi:  # 看是否超过最大未更新最优适应度轮数
                    pop = pop[pop[:, self.dim].argsort()]  # 按照适应度升序
                    if pop[0, self.dim] >= self.best_function:
                        t = self.iterate+1
                        time.sleep(2)
                        self.stop_mniwi = True

        # Sort
        pop = pop[pop[:, self.dim].argsort()]  # 升序

        # 输出最优目标函数和最优解（可行解）
        if self.eq_constraints is None and self.ineq_constraints is None:  # 无约束条件
            if pop[0, self.dim] < self.best_function:
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, : self.dim].copy()
        else:  # 有约束条件
            for i in range(len(pop)):
                if pop[i, self.dim] < self.best_function and self.penalty(pop[i, :self.dim]) <= 0:
                    self.best_function = pop[i, self.dim].copy()
                    self.best_variable = pop[i, : self.dim].copy()
                    break

        # 记录当前最优解Report
        self.report.append(pop[0, self.dim])

        # 输出字典
        self.output_dict = {'variable': self.best_variable, 'function': \
            self.best_function}
        if self.progress_bar == True:
            show = ' ' * 100
            sys.stdout.write('\r%s' % (show))
        judge = '局部最优解是可行解' if self.penalty(self.best_variable) <= 0 else '该局部最优解不可行，找不到可行的局部最优解'
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush()
        print(f'{judge}')
        re = np.array(self.report)

        if self.convergence_curve == True:  # 绘制收敛曲线
            plt.figure(figsize=(8, 6))
            plt.tick_params(size=5, labelsize=13)  # 坐标轴
            plt.grid(alpha=0.3)  # 是否加网格线
            plt.plot(np.arange(t), re, color='#e74c3c', lw=1.5)
            plt.xlabel('Iteration', fontsize=13)
            plt.ylabel('Objective function value', fontsize=13)
            plt.title('Genetic Algorithm', fontsize=15)
            if self.plot_path is not None:
                plt.savefig(f'{self.plot_path}/Genetic Algorithm.svg', format='svg', bbox_inches='tight')
            plt.show()

        if self.stop_mniwi == True:  # 退出循环
            sys.stdout.write('\nWarning: GA is terminated due to the' + \
                             ' maximum number of iterations without improvement was met!')

    def penalty(self, X):  # 根据所有约束条件计算惩罚项
        if self.eq_constraints is not None:  #有等式约束时
            eq_cons_output = self.eq_constraints(X)  # 输出约束条件值
            #将等式约束拆分成两个不等式约束（小于等于）
            eq_cons_output= [output-self.eq_cons_coefficient-self.eq_cons_coefficient*2*i
                             for i in range(2)
                             for output in eq_cons_output]
            eq_conpare_matrix = np.zeros((len(eq_cons_output), 2))
            eq_conpare_matrix[:, 1] = eq_cons_output
            eq_penalty=self.penalty_factor * np.sum(np.max(eq_conpare_matrix, axis=1))
        else:
            eq_penalty=0

        if self.ineq_constraints is not None:  #有不等式约束时
            ineq_cons_output = self.ineq_constraints(X)  # 输出约束条件值
            ineq_conpare_matrix = np.zeros((len(ineq_cons_output), 2))
            ineq_conpare_matrix[:, 1] = ineq_cons_output
            ineq_penalty=self.penalty_factor * np.sum(np.max(ineq_conpare_matrix, axis=1))
        else:
            ineq_penalty=0

        if self.eq_constraints is not None or self.ineq_constraints is not None:  # 没有约束条件时
            return eq_penalty+ineq_penalty
        else:
            return None

    def cross(self, x, y, c_type):  # 交叉

        ofs1 = x.copy()
        ofs2 = y.copy()

        # 这里的遗传算法是通过直接调换两个解的对应顺序
        if c_type == 'one_point':  # 单点交叉
            ran = np.random.randint(0, self.dim)  # 选择点位
            ofs1[:ran] = y[:ran].copy()
            ofs2[:ran] = x[:ran].copy()

        if c_type == 'two_point':
            # 随机选择两个点位
            ran1 = np.random.randint(0, self.dim)
            ran2 = np.random.randint(ran1, self.dim)
            # 进行交叉（交换）
            ofs1[ran1:ran2] = y[ran1:ran2].copy()
            ofs2[ran1:ran2] = x[ran1:ran2].copy()

        if c_type == 'uniform':  # 均匀交叉

            for i in range(0, self.dim):
                ran = np.random.random()
                if ran < 0.5:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

        return np.array([ofs1, ofs2])  # 返回两个子代解

    def mut(self, x):  # 变异（对每个决策变量都有小的概率重新生成一个在值域范围内的值）

        for i in self.integers[0]:  # 对整数决策变量
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = np.random.randint(self.var_bound[i][0], \
                                         self.var_bound[i][1] + 1)

        for i in self.reals[0]:  # 对连续的决策变量
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = self.var_bound[i][0] + np.random.random() * \
                       (self.var_bound[i][1] - self.var_bound[i][0])

        return x  # 返回解

    def mutmidle(self, x, p1, p2):  # 在两个父代解中进行变异，值域限制在父代两个解之间
        for i in self.integers[0]:  # 对整数变量
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = np.random.randint(p1[i], p2[i])
                elif p1[i] > p2[i]:
                    x[i] = np.random.randint(p2[i], p1[i])
                else:
                    x[i] = np.random.randint(self.var_bound[i][0], \
                                             self.var_bound[i][1] + 1)

        for i in self.reals[0]:  # 对连续变量
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = p1[i] + np.random.random() * (p2[i] - p1[i])
                elif p1[i] > p2[i]:
                    x[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
                else:
                    x[i] = self.var_bound[i][0] + np.random.random() * \
                           (self.var_bound[i][1] - self.var_bound[i][0])
        return x

    def evaluate(self):  # 评估适应度(加入惩罚)
        return self.f(self.temp) + self.penalty(self.temp)

    def sim(self, X):  # 嵌套了测试目标函数是否可用的一个功能，实际还是返回目标函数
        self.temp = X.copy()
        obj = None
        try:
            obj = func_timeout(self.funtimeout, self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        assert (obj != None), "After " + str(self.funtimeout) + " seconds delay " + \
                              "func_timeout: the given function does not provide any output"
        return obj
