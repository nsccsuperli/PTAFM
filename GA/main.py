import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import numpy as np
from pprint import pprint
import time
import math
import copy
from tqdm import tqdm
from data.ACO_Solutions import *
from cost import *
import ga
import re
import os
import RegExService

if not os.path.exists('results'):
    os.makedirs('results')

# 参数定义
num_tests = 10  # number of iid tests
dp_comm_size = 1000  # 数据并行通信量
pp_comm_size = 1000  # 流水线并行通信量
n_pop = 100  # 种群大小
r_mut1 = 0  # 中心翻转突变概率
r_mut2 = 0.6  # stage内变异概率
r_mut3 = 0.5  # stage间交换概率
dp_weight = 1  # 数据并行权重
pp_weight = 2  # 流水线并行权重
sd_weight = 5  # 算力平衡权重
term_count = 1000  # 终止条件

# 获取数据
power, devices, latency, bandwidth, num_stage = RegExService.getData('data/data-center-config.txt')
n_idc = n = len(power)
# 延迟矩阵
latency_mat = np.zeros([n, n])
for (row, col), value in latency.items():
    latency_mat[row - 1, col - 1] = value
for i in range(1, n):
    for j in range(i):
        latency_mat[i, j] = latency_mat[j, i]
# 带宽矩阵
bandwidth_mat = np.zeros([n, n])
for (row, col), value in bandwidth.items():
    bandwidth_mat[row - 1, col - 1] = value
for i in range(1, n):
    for j in range(i):
        bandwidth_mat[i, j] = bandwidth_mat[j, i]

result = {'best_sol': [], 'best_cost': math.inf, 'cost': [0] * num_tests, 'avg_cost': math.inf, 'cost_std': math.inf}
best_cost = math.inf
best_sol = []
data = {}

# run and visualization
for _ in tqdm(range(num_tests)):
    best_sol, best_cost, best_path, data = ga.ga(n, latency_mat, bandwidth_mat, candidate_solutions, dp_comm_size,
                                                 pp_comm_size, n_pop, r_mut1, r_mut2, r_mut3, term_count, power,
                                                 capacityLimit, num_stage, dp_weight, pp_weight, sd_weight)

    result['cost'][_] = best_cost
    if best_cost < result['best_cost']:
        result['best_sol'] = best_sol
        result['best_cost'] = best_cost
        result['best_path'] = best_path.copy()
    # plt.plot(range(len(data['cost'])), data['cost'], color='b', alpha=math.pow(num_tests, -0.75))
    plt.plot(range(len(data['cost'])), data['best_cost'], color='r', alpha=math.pow(num_tests, -0.75))

plt.title('Solving TSP with Genetic Algorithm')
plt.xlabel('Number of Iteration')
plt.ylabel('Cost')
plt.savefig('results/ga.png')

# print results
result['avg_cost'] = np.mean(result['cost'])
result['cost_std'] = np.std(result['cost'])

pprint(result)

# 结果排序
solution = [[] for _ in range(len(best_path))]
for index, _ in enumerate(best_path):
    solution[_] = best_sol[index]

# 转换为设备节点的准备工作
content = open("data/data-center-config.txt", "r").read()
power_data = re.findall(r"^(\d+) \[(\d+),(\d+)\]", content, re.MULTILINE)
idc_devices_dict = {int(node): [int(a), int(t)] for node, a, t in [line for line in power_data]}

index = 0
idc_nodes_dict = {int(node): [[], []] for node, _, _ in [line for line in power_data]}
for idc in range(1, n_idc + 1):
    for a in range(1, idc_devices_dict[idc][0] + 1):
        idc_nodes_dict[idc][0].append(index)
        index += 1

for idc in range(1, n_idc + 1):
    for t in range(1, idc_devices_dict[idc][1] + 1):
        idc_nodes_dict[idc][1].append(index)
        index += 1


# print(idc_nodes_dict)

def convert_to_node(solution, completePower, remainingPower, split_idc):
    idc_nodes = copy.deepcopy(idc_nodes_dict)
    solution_devices = []
    for stage in solution:
        temp_devices = []
        if len(stage) > 1:  # 多个数据中心组合
            for idc in stage:
                if idc not in split_idc:  # 多个数据中心作为一个stage
                    temp_devices.extend(idc_nodes[idc][0])
                    temp_devices.extend(idc_nodes[idc][1])
                    idc_nodes[idc][0] = []
                    idc_nodes[idc][1] = []
                elif idc in split_idc and idc in remainingPower.keys():  # 大数据中心拆分出一个小的stage的部分
                    for _ in range(remainingPower[idc][0]):
                        temp_devices.append(idc_nodes[idc][0].pop(0))
                    for _ in range(remainingPower[idc][1]):
                        temp_devices.append(idc_nodes[idc][1].pop(0))
        elif len(stage) == 1:  # 单数据中心
            idc = stage[0]
            if idc not in split_idc:  # 中型数据中心单独作为一个stage
                temp_devices.extend(idc_nodes[stage[0]][0])
                temp_devices.extend(idc_nodes[stage[0]][1])
                idc_nodes[stage[0]][0] = []
                idc_nodes[stage[0]][1] = []
                pass
            elif idc in split_idc:  # 大数据中心拆分出的一个完整的stage的部分
                for allocation in completePower[idc]:
                    for _ in range(allocation[0]):
                        temp_devices.append(idc_nodes[idc][0].pop(0))
                    for _ in range(allocation[1]):
                        temp_devices.append(idc_nodes[idc][1].pop(0))

        solution_devices.append(temp_devices)
    return solution_devices


final_solution = convert_to_node(solution, completePower, remainingPower, search_vertices)
print('数据中心节点分布：')
print(idc_nodes_dict)
print('最终解决方案：')
print(final_solution)
