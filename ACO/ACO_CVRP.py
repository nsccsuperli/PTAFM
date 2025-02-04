import re
import sys
import copy
import math
import getopt
import random
import itertools
import numpy as np
from unittest import result
from functools import reduce

ro = 0.8  # 信息素保留率 0.8
th = 80  # 每次释放信息素的量
ants = 20  # 每次迭代生成的方案
alfa = 2  # 信息素重要程度因子 2
beta = 5  # 启发函数重要程度因子 5
sigm = 3  # 每次迭代的最优解决进行局部更新的信息素的量 3
lbd1 = 1  # 距离的权重
lbd2 = 2  # 标准差的权重
fileName = "data-center-config.txt"
iterations = 200
communicate_size = 1000  # 通讯量大小：距离=延迟+通讯量/带宽
min_capacity_ratio = 0.9  # 每个Stage最小的算力比率 80%


def getData(fileName):
    f = open(fileName, "r")
    content = f.read()
    # capacity = re.search("^CAPACITY:(\d+)$", content, re.MULTILINE).group(1)  # 6000
    num_stage = re.search("^NUM_STAGE:(\d+)$", content, re.MULTILINE).group(1)
    # min_capacity_ratio = re.search("^MIN_CAPACITY_RATIO:(\d+\.\d+)$", content, re.MULTILINE).group(1)
    power_data = re.findall(r"^(\d+) \[(\d+),(\d+)\]", content, re.MULTILINE)
    latency_data = re.search(r'LATENCY_SECTION\n(.*?)\nend', content, re.DOTALL).group(1).split('\n')
    bandwidth_data = re.search(r'BANDWIDTH_SECTION\n(.*?)\nend', content, re.DOTALL).group(1).split('\n')

    power_dict = {int(node): int(int(a) * 7 + int(t)) for node, a, t in [line for line in power_data]}
    devices_dict = {int(node): [int(device1), int(device2)] for node, device1, device2 in
                    [line for line in power_data]}
    latency_dict = {(float(node1), float(node2)): float(latency) for node1, node2, latency in
                    [line.split() for line in latency_data]}
    bandwidth_dict = {(float(node1), float(node2)): float(bandwidth) for node1, node2, bandwidth in
                      [line.split() for line in bandwidth_data]}
    # capacity = int(capacity)
    num_stage = int(num_stage)
    # min_capacity_ratio = float(min_capacity_ratio)
    return power_dict, devices_dict, latency_dict, bandwidth_dict, num_stage


def generateGraph():
    power, devices, latency, bandwidth, num_stage = getData(fileName)
    vertices = list(power.keys())
    # vertices.remove(1)  # 城市1作为仓库
    # edges={(1, 1): 0.0, (1, 2): 49.36598018878993, (1, 3): 48.08326112068523, ...
    edges = {
        (min(a, b), max(a, b)): latency[(min(a, b), max(a, b))] + communicate_size / (
                bandwidth[(min(a, b), max(a, b))] * 1e3 / 8)
        for a in power.keys() for b in power.keys()}
    feromones = {(min(a, b), max(a, b)): 1 for a in power.keys() for b in power.keys() if a != b}  # 所有路径上的信息素初始化为1
    all_power = sum(v for v in power.values())
    capacityLimit = all_power / num_stage
    return vertices, edges, capacityLimit, power, devices, feromones, num_stage, all_power


def solutionOfOneAnt(vertices, edges, capacityLimit, min_capacity_ratio, power, feromones):
    solution = list()

    while (len(vertices) != 0):
        path = list()
        city = np.random.choice(vertices)  # 随机选择一个城市作为开始
        capacity = capacityLimit - power[city]  # 剩余容量=车的容量-城市需求
        path.append(city)  # 车走过的的路径增加
        vertices.remove(city)  # 从车的待访问路径中删除
        while (len(vertices) != 0):  # 该城市到其他城市的概率
            probabilities = list(map(lambda x: ((feromones[(min(x, city), max(x, city))]) ** alfa) * (
                    (1 / edges[(min(x, city), max(x, city))]) ** beta), vertices))
            probabilities = probabilities / np.sum(probabilities)

            city = np.random.choice(vertices, p=probabilities)  # 轮盘赌的方式选择下一个城市

            if (capacity > (1 - min_capacity_ratio) * capacityLimit):  # 若剩余容量大于 0.2*stage_power
                capacity = capacity - power[city]  # 剩余容量可能是负值（车含有的算力超出stage_power）
                path.append(city)  # 城市加入车走过的的路径
                vertices.remove(city)  # 从车的路径选择中删除
            else:
                break
        solution.append(path)  # 存储车的路径
    return solution


def rateSolution(solution, edges, power, capacityLimit, num_stage):
    """统计该解决方案的成本(路径长度及标准差)"""
    sd = 0  # 算力标准差
    s = []  # stage内通讯的最大值的列表
    for i in solution:  # 每辆车走的路径  i=[5, 7, 2]
        sd_ = 0
        s_ = []  # 每个stage中当前节点与其他节点的通讯
        for j in i:
            comm = 0
            sd_ = sd_ + power[j]
            for k in i:
                if j != k:
                    comm = comm + edges[(min(j, k), max(j, k))]
            s_.append(comm)
        s.append(max(s_))
        sd = sd + (sd_ - capacityLimit) ** 2
    sd = math.sqrt(sd / num_stage)
    return lbd1 * sum(s) + lbd2 * sd


def updateFeromone(feromones, solutions, bestSolution):
    Lavg = reduce(lambda x, y: x + y, (i[1] for i in solutions)) / len(solutions)  # 计算本次迭代22个解决方案的平均路径长度
    feromones = {k: (ro + th / Lavg) * v for (k, v) in feromones.items()}  # 原信息素更新方式
    # feromones = {k: ro * v + th / Lavg for (k, v) in feromones.items()}  # 新
    solutions.sort(key=lambda x: x[1])  # 按照成本排序

    if bestSolution != None:
        if solutions[0][1] < bestSolution[1]:  # 根据成本找到最佳的方案
            bestSolution = solutions[0]

        for path in bestSolution[0]:  # 为最优的解决方案更新信息素
            for i in range(len(path) - 1):
                # feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = sigm/bestSolution[1] + feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
                feromones[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))] = 10 / bestSolution[1] + feromones[
                    (min(path[i], path[i + 1]), max(path[i], path[i + 1]))]
    else:
        bestSolution = solutions[0]

    for l in range(sigm):  # 对每次迭代中的前sigm个车的路线进行信息素更新
        paths = solutions[l][0]
        L = solutions[l][1]
        for path in paths:
            for i in range(len(path) - 1):
                # 原更新方式
                # feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = (sigm-(l+1)/L**(l+1)) + feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
                feromones[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))] = (sigm - (l + 1) / L ** (l + 1)) + \
                                                                                    feromones[(
                                                                                        min(path[i], path[i + 1]),
                                                                                        max(path[i], path[i + 1]))]

    return bestSolution


def search_big_datacenter(vertices, power, devices, capacityLimit):
    completePower = {}  # 数据中心的完整部分
    remainingPower = {}  # 该数据中心的完整部分
    num_big = 0  # 完整部分的数量

    for vertice in vertices.copy():
        used_devices = []  # 每个节点的完整部分的设备情况
        if power[vertice] >= capacityLimit:

            while power[vertice] >= capacityLimit:
                power_sum = 0  # 当前找到的算力和
                use_devices = [0, 0]  # 当前使用的设备
                proto_devices = devices[vertice].copy()  # 原始的设备情况

                # 直到A100不足（性能不足），或者A100剩余（性能过剩）
                while power_sum < capacityLimit:
                    if devices[vertice][0]:  # 使用A100
                        power_sum = power_sum + 7
                        devices[vertice][0] = devices[vertice][0] - 1
                        use_devices[0] = use_devices[0] + 1
                    else:
                        break

                # 若A100不足，增加T4
                while power_sum < capacityLimit:
                    if devices[vertice][1]:
                        power_sum = power_sum + 1
                        devices[vertice][1] = devices[vertice][1] - 1
                        use_devices[1] = use_devices[1] + 1

                # 如果找到的资源性能过剩
                if power_sum > capacityLimit:
                    # 其不包含T4，暂存目前的 算力和、当前使用设备
                    if not use_devices[1]:  # 如果不含有T4
                        temp_sum = power_sum
                        temp_use_devices = use_devices.copy()
                        # 若去除一个A100，剩余T4可以补充
                        if devices[vertice][1] > capacityLimit - power_sum + 7:
                            power_sum = power_sum - 7
                            devices[vertice][0] = devices[vertice][0] + 1
                            use_devices[0] = use_devices[0] - 1
                            while power_sum < capacityLimit:
                                if devices[vertice][1]:
                                    power_sum = power_sum + 1
                                    devices[vertice][1] = devices[vertice][1] - 1
                                    use_devices[1] = use_devices[1] + 1
                                else:
                                    break
                        if temp_sum <= power_sum:  # 全A100用的资源更少
                            power[vertice] = power[vertice] - temp_sum
                            devices[vertice] = [proto_devices[0] - temp_use_devices[0],
                                                proto_devices[1] - temp_use_devices[1]]
                        else:
                            power[vertice] = power[vertice] - power_sum
                    else:
                        power[vertice] = power[vertice] - power_sum

                if power_sum == capacityLimit:
                    power[vertice] = power[vertice] - power_sum
                used_devices.append(use_devices)

            if power[vertice]:  # 如果节点仍有算力
                remainingPower[vertice] = devices[vertice]
            else:
                vertices.remove(vertice)
        if used_devices:
            completePower[vertice] = used_devices
            num_big = num_big + len(used_devices)  # 统计大数据中心划分出来的完整的Stage的数量

    return completePower, num_big, remainingPower, vertices, power, devices


def search_middle_datacenter(vertices, power, capacityLimit):
    middle = []
    for vertice in vertices:
        if min_capacity_ratio * capacityLimit < power[vertice] < capacityLimit:
            middle.append(vertice)
            vertices.remove(vertice)

    return middle, len(middle), vertices


def judging_neighbors(completePower, remainingPower, candidates):
    candidates = [tup[0] for tup in candidates]  # 只保留节点编号

    split_idc = []
    for key in completePower.keys():  # 找到拆分的数据中心
        if key not in split_idc and (len(completePower[key]) >= 2 or key in list(remainingPower)):
            split_idc.append(key)

    result = candidates.copy()
    for candidate in candidates:  # 对每个组合判断数据中心是否相邻
        for vertex in split_idc:
            if candidate in result:
                point_position = []
                for _, stage in enumerate(candidate):
                    if vertex in stage:
                        point_position.append(_)
                for i in range(len(point_position) - 1):
                    if point_position[i + 1] - point_position[i] != 1:
                        result.remove(candidate)
                        break
            else:
                break
    return result, split_idc


def save_file(candidates, completePower, remainingPower, capacityLimit, split_idc,idc_nodes_dict):
    file_path = "ACO_Solutions.py"
    with open(file_path, 'w') as file:
        file.write('"""ACO_Solutions"""')
        file.write('\ncandidate_solutions = ' + str(list(candidates)))
        file.write('\ncompletePower = ' + str(completePower))
        file.write('\nremainingPower = ' + str(remainingPower))
        file.write('\ncapacityLimit = ' + str(capacityLimit))
        file.write('\nsplit_idc = ' + str(split_idc))
        file.write('\nidc_nodes_dict = ' + str(idc_nodes_dict))
    print("ACO算法生成的种群已成功保存到", file_path)



def idcs_to_nodes(fileName, power, candidate_solutions, completePower, remainingPower, split_idc):
    n_idc = len(power)

    content = open(fileName, "r").read()
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
    print('数据中心含有的设备编号：')
    print(idc_nodes_dict)

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

    candidate_node_solutions = []
    for candidate_solution in candidate_solutions:
        candidate_node_solution = convert_to_node(candidate_solution, completePower, remainingPower, split_idc)
        candidate_node_solutions.append(candidate_node_solution)

    return candidate_node_solutions,idc_nodes_dict


def main():
    bestSolution = None
    vertices, edges, capacityLimit, power, devices, feromones, num_stage, all_power = generateGraph()
    # 大数据中心的拆分
    completePower, num_big, remainingPower, vertices, power, devices = search_big_datacenter(vertices, power, devices,
                                                                                             capacityLimit)
    # 中数据中心的查找
    middle, num_middle, vertices = search_middle_datacenter(vertices, power, capacityLimit)
    # 小数据中心的组合
    candidate_solutions = list()
    candidates = list()

    for i in range(iterations):  # 迭代次数
        solutions = list()
        for _ in range(ants):  # 每次迭代，每只蚂蚁都会生成一种方案
            solution = solutionOfOneAnt(vertices.copy(), edges, capacityLimit, min_capacity_ratio, power, feromones)
            cost = rateSolution(solution, edges, power, capacityLimit, num_stage)
            solution.extend([[k] for k, v in completePower.items() for _ in v])
            solution.extend([[v] for v in middle])
            if len(solution) == num_stage:
                random.shuffle(solution)
                solutions.append((solution, cost))  # 每次迭代的方案及其成本汇总
        # 若solution为空，提示错误信息，退出程序
        if not solutions:
            print("\033[31mError:无法对输入的图进行搜索！请检查输入的NUM_STAGE及min_capacity_ratio是否合理！\033[0m\n")
            print("\033[31m提示:\033[0m")
            if len(solution) > num_stage:
                print("当前方案可能存在多余的算力，参考值:", solution)
            if len(solution) < num_stage:
                print("当前方案算力可能无法满足最低要求，参考值:", solution)
                print("Complete section:", completePower, "\nRemaining section:", remainingPower)
            exit()

        candidate_solutions.extend(solutions)
        bestSolution = updateFeromone(feromones, solutions, bestSolution)  # 每次迭代的最佳方案
        print(str(i) + ": " + str(bestSolution[1]))

    [candidates.append(i) for i in candidate_solutions if not i in candidates]  # 候选结果去重
    candidates.sort(key=lambda x: x[1])  # 候选排序

    candidates, split_idc = judging_neighbors(completePower, remainingPower, candidates)
    candidate_node_solutions,idc_nodes_dict = idcs_to_nodes(fileName, power, candidates[:100], completePower, remainingPower,
                                             split_idc)
    save_file(candidate_node_solutions, completePower, remainingPower, capacityLimit, split_idc,idc_nodes_dict)

    return bestSolution, completePower, remainingPower, capacityLimit, all_power


if __name__ == "__main__":
    bestSolution, completePower, remainingPower, capacityLimit, all_power = main()
    print("AllPower:", all_power)
    print("PerStagePower:", capacityLimit)
    print("BestSolution: " + str(bestSolution))
    print("Big data center computing power equipment:\n" + "Complete section:", completePower, "\nRemaining section:",
          remainingPower)
