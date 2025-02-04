import math
import numpy as np
import itertools


def get_cost(n, latency_mat, bandwidth_mat, solution, power, capacityLimit, n_stage, dp_comm_size, pp_comm_size,
             dp_weight, pp_weight, sd_weight, search_vertices, remainingPower, completePower):
    """从解决方案中获取的路径总成本"""

    def dp_cost():
        """数据并行成本"""
        s = 0  # 路径的最大值
        for i in solution:  # 每辆车走的路径  i=[5, 7, 2]
            s_ = 0
            if len(i) > 1:  # Stage的数据中心个数大于1
                combinations = list(itertools.combinations(i, 2))
                for k in combinations:
                    s_ = max(s_, latency_mat[k[0] - 1][k[1] - 1] + dp_comm_size / bandwidth_mat[k[0] - 1][k[1] - 1])
            s = s + s_
        return s

    def pp_cost():
        """流水线并行成本"""

        class open_loop_tsp:
            """一个基于动态规划的开环旅行商问题（TSP）求解器"""

            def __init__(self, cost_matrix, start_node):
                self.cost_matrix = cost_matrix
                self.num_nodes = self.cost_matrix.shape[0]
                self.start_node = start_node
                self.dp_table = np.full(
                    shape=(self.num_nodes, pow(2, self.num_nodes)), fill_value=np.inf)
                self.trace_table = np.zeros(
                    shape=(self.num_nodes, pow(2, self.num_nodes)))

            def convert(self, future_nodes):
                binary_future_nodes = 0
                for future_node in future_nodes:
                    binary_future_nodes += pow(2, future_node)
                return binary_future_nodes

            def solve(self, node, future_nodes):
                if len(future_nodes) == 0:
                    # closed loop tsp problem: return self.cost_matrix[node][self.start_node]
                    # open loop tsp problem: return 0
                    return 0

                all_distance = []
                for next_node in future_nodes:
                    next_future_nodes = future_nodes.copy()
                    next_future_nodes.remove(next_node)
                    binary_next_future_nodes = self.convert(next_future_nodes)
                    if self.dp_table[next_node][binary_next_future_nodes] == np.inf:
                        all_distance.append(
                            self.cost_matrix[node][next_node] + self.solve(next_node, next_future_nodes))
                    else:
                        all_distance.append(
                            self.cost_matrix[node][next_node] + self.dp_table[next_node][binary_next_future_nodes])

                min_distance = min(all_distance)
                next_node = future_nodes[all_distance.index(min_distance)]

                binary_future_nodes = self.convert(future_nodes)
                self.dp_table[node][binary_future_nodes] = min_distance
                self.trace_table[node][binary_future_nodes] = next_node
                return min_distance

            def get_least_cost_route(self):
                future_nodes = list(range(self.num_nodes))
                future_nodes.remove(self.start_node)
                cost = self.solve(self.start_node, future_nodes)

                path = [self.start_node]
                cur_node = self.start_node
                while len(future_nodes) > 0:
                    binary_future_nodes = self.convert(future_nodes)
                    cur_node = int(self.trace_table[cur_node][binary_future_nodes])
                    future_nodes.remove(cur_node)
                    path.append(cur_node)
                return cost, path

        def comm_cost(a, b):
            stage1 = a.copy()
            stage2 = b.copy()
            cost = 0
            sum1 = 0
            sum2 = 0

            if len(stage1) == 1:
                power1 = [power[stage1[0]] if stage1[0] not in search_vertices else \
                              completePower[stage1[0]][0][0] * 7 + completePower[stage1[0]][0][1]]
            else:
                power1 = [power[node] if node not in search_vertices else \
                              remainingPower[node][0] * 7 + remainingPower[node][1] for node in stage1]

            if len(stage2) == 1:
                power2 = [power[stage2[0]] if stage2[0] not in search_vertices else \
                              completePower[stage2[0]][0][0] * 7 + completePower[stage2[0]][0][1]]
            else:
                power2 = [power[node] if node not in search_vertices else \
                              remainingPower[node][0] * 7 + remainingPower[node][1] for node in stage2]

            index1 = stage1.pop(0)
            sum1 += power1.pop(0)
            index2 = stage2.pop(0)
            sum2 += power2.pop(0)
            cost = max(cost, latency_mat[index1 - 1, index2 - 1] + pp_comm_size / bandwidth_mat[index1 - 1, index2 - 1])

            while stage1 or stage2:
                while sum1 > sum2 and stage2:
                    index2 = stage2.pop(0)
                    sum2 += power2.pop(0)
                    cost = max(cost, latency_mat[index1 - 1, index2 - 1] + \
                               pp_comm_size / bandwidth_mat[index1 - 1, index2 - 1])
                while sum2 >= sum1 and stage1:
                    index1 = stage1.pop(0)
                    sum1 += power1.pop(0)
                    cost = max(cost, latency_mat[index1 - 1, index2 - 1] + \
                               pp_comm_size / bandwidth_mat[index1 - 1, index2 - 1])
                if (sum1 > sum2 and stage1) or (sum2 >= sum1 and stage2):
                    break

            return cost

        cross_partition_cost = np.zeros(shape=(n_stage, n_stage))

        sol = solution.copy()
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                cross_partition_cost[i][j] = comm_cost(sol[i], sol[j])
                cross_partition_cost[j][i] = cross_partition_cost[i][j]

        pipeline_parallel_cost = []
        pipeline_parallel_path = []
        for start_node in range(n_stage):
            tsp = open_loop_tsp(cross_partition_cost, start_node)  # 在所有起始节点上运行 open_loop_tsp，计算了管道并行的最小成本和最优路径。
            cost, path = tsp.get_least_cost_route()
            pipeline_parallel_cost.append(cost)
            pipeline_parallel_path.append(path)
        dp_pipeline_parallel_cost = min(pipeline_parallel_cost)
        dp_pipeline_parallel_path = pipeline_parallel_path[pipeline_parallel_cost.index(dp_pipeline_parallel_cost)]

        return dp_pipeline_parallel_cost, dp_pipeline_parallel_path

    def power_cost():
        """算力平衡性"""
        sd = 0  # 算力标准差
        for i in solution:  # 每辆车走的路径  i=[5, 7, 2]
            sd_ = 0
            for j in i:
                sd_ = sd_ + power[j]
            sd = sd + (sd_ - capacityLimit) ** 2
        sd = math.sqrt(sd / n_stage)
        return sd

    dp_cost = dp_cost()
    pp_cost, pp_path = pp_cost()
    power_cost = power_cost()

    return dp_weight * dp_cost + pp_weight * pp_cost + sd_weight * power_cost, pp_path
