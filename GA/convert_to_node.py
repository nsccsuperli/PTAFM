from data.ACO_Solutions import *
import copy
import re

n_idc = 7

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



# candidate_node_solution = convert_to_node(candidate_solution, completePower, remainingPower, search_vertices)

