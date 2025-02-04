import numpy as np
from cost import get_cost
import random
import copy
from collections import deque
from data.ACO_Solutions import *


def tnm_selection(pop, costs, max_tnm):  # tournament selection
    """
    实施基于锦标赛选择(tournament selection)的个体选择操作。

    参数:
    pop - 一个列表，包含了所有待选择的个体。
    costs - 一个列表，对应于pop中每个个体的适应度成本。
    max_tnm - 一个整数，表示锦标赛中参与竞争的个体数量。

    返回:
    返回在锦标赛选择中胜出的个体。
    """
    selection_ix = np.random.randint(0, len(pop))
    for _ in np.random.randint(0, len(pop), max_tnm - 1):  # 随机找到3个个体 [0, len(pop)-1]
        # check if better (e.g. perform a tournament)
        if costs[_] < costs[selection_ix]:
            selection_ix = _
    return pop[selection_ix]


def crossover(p_1, p_2, r_cross):
    """交叉 （没有加入拆分的数据中心必须相邻的限制）"""
    if random.random() < r_cross:  # 生成一个0到1之间的随机数
        c1, c2 = copy.deepcopy(p_1), copy.deepcopy(p_2)
        # 判断两个父代是否可以交叉
        a = [set(sublist) for sublist in p_1]
        b = [set(sublist) for sublist in p_2]
        aa = copy.deepcopy(a)
        bb = copy.deepcopy(b)
        for i in a:
            if i not in b:
                return [p_1, p_2]
        # 找到两个交叉点
        pt_1 = random.randint(0, len(p_1) - 1)
        pt_2 = random.randint(0, len(p_2) - 1)
        while pt_1 == pt_2:
            pt_2 = random.randint(0, len(p_2) - 1)
        pt_1, pt_2 = (pt_1, pt_2) if pt_1 < pt_2 else (pt_2, pt_1)
        # 通过集合找到两个父代需要删除的位置

        for _ in range(pt_2 - pt_1):
            index1 = a.index(bb[pt_1 + _])
            index2 = b.index(aa[pt_1 + _])
            a.pop(index1)
            c1.pop(index1)
            b.pop(index2)
            c2.pop(index2)

        temp1 = c1[:pt_1] + p_2[pt_1:pt_2] + c1[pt_1:]
        temp2 = c2[:pt_1] + p_1[pt_1:pt_2] + c2[pt_1:]
        return [temp1, temp2]
    else:
        return [p_1, p_2]


def mutation1(parent, r_mut):
    p = copy.deepcopy(parent)
    """中心翻转突变 center inverse mutation / CIM"""
    if random.random() < r_mut:
        index = random.randint(1, len(p) - 1)
        list1 = p[:index]
        list2 = p[index:]
        return list1[::-1] + list2[::-1]
    else:
        return p


def mutation2(parent, r_mut):
    p = copy.deepcopy(parent)
    """一个stage内交换数据中心"""
    if random.random() < r_mut:
        pt = random.randint(0, len(p) - 1)  # 选择某个Stage
        while len(p[pt]) <= 1:
            pt = random.randint(0, len(p) - 1)
        pt_1 = random.randint(0, len(p[pt]) - 1)
        pt_2 = random.randint(0, len(p[pt]) - 1)
        while pt_1 == pt_2:
            pt_2 = random.randint(0, len(p[pt]) - 1)
        temp = p[pt][pt_1]
        p[pt][pt_1] = p[pt][pt_2]
        p[pt][pt_2] = temp

    return p


def mutation3(parent, r_mut):
    """两个stage间交换数据中心"""
    p = copy.deepcopy(parent)
    if random.random() < r_mut:
        stage_index = []
        for _, stage in enumerate(p):
            if len(stage) > 1 and set(stage) != set(search_vertices):
                stage_index.append(_)
        if len(stage_index) < 2:
            return p

        stage1, stage2 = random.sample(stage_index, 2)
        a = random.randint(0, len(p[stage1]) - 1)
        while p[stage1][a] in search_vertices:
            a = random.randint(0, len(p[stage1]) - 1)
        b = random.randint(0, len(p[stage2]) - 1)
        while p[stage2][b] in search_vertices:
            b = random.randint(0, len(p[stage2]) - 1)

        temp = p[stage1][a]
        p[stage1][a] = p[stage2][b]
        p[stage2][b] = temp

    return p


def ga(n, latency_mat, bandwidth_mat, candidate_solutions, dp_comm_size, pp_comm_size, n_pop, r_mut1, r_mut2,
       r_mut3, term_count, power, capacityLimit, num_stage, dp_weight, pp_weight, sd_weight):

    pop = copy.deepcopy(candidate_solutions)  # 获取ACO生成n_pop个种群

    data = {'cost': deque([]), 'best_cost': deque([])}  # cost 表示平均成本
    count = 0
    costs = []
    pp_paths = []
    for _ in pop:
        a, b = get_cost(n, latency_mat, bandwidth_mat, _, power, capacityLimit, num_stage, dp_comm_size, pp_comm_size,
                        dp_weight, pp_weight, sd_weight, search_vertices, remainingPower, completePower)
        costs.append(a)
        pp_paths.append(b)

    while True:
        min_index = np.argmin(costs)
        best_cost = costs[min_index]
        best_sol = copy.deepcopy(pop[min_index])
        best_path = copy.deepcopy(pp_paths[min_index])
        data['cost'].append(np.mean(costs))
        data['best_cost'].append(best_cost)
        count += 1
        if count > term_count:
            return best_sol, best_cost, best_path, data

        np.random.seed = count  # 设置随机性种子保证随机性
        parent1_idx, parent2_idx = np.random.randint(n_pop, size=2)  # 随机生成两个父代的索引

        p1, p2 = copy.deepcopy(pop[parent1_idx]), copy.deepcopy(pop[parent2_idx])  # get selected parents in pairs
        for c in (p1, p2):
            # c = mutation1(c, r_mut1)
            c = mutation2(c, r_mut2)
            c = mutation3(c, r_mut3)

            offspring_score, pp_path = get_cost(n, latency_mat, bandwidth_mat, c, power, capacityLimit, num_stage,
                                                dp_comm_size, pp_comm_size, dp_weight, pp_weight, sd_weight,
                                                search_vertices, remainingPower, completePower)
            replaced_idx = parent1_idx if costs[parent1_idx] > costs[parent2_idx] else parent2_idx
            if offspring_score <= costs[replaced_idx]:
                pop[replaced_idx] = copy.deepcopy(c)
                costs[replaced_idx] = offspring_score
                pp_paths[replaced_idx] = copy.deepcopy(pp_path)
