import re


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
    latency_dict = {(int(node1), int(node2)): int(latency) for node1, node2, latency in
                    [line.split() for line in latency_data]}
    bandwidth_dict = {(int(node1), int(node2)): int(bandwidth) for node1, node2, bandwidth in
                      [line.split() for line in bandwidth_data]}
    # capacity = int(capacity)
    num_stage = int(num_stage)
    # min_capacity_ratio = float(min_capacity_ratio)
    return power_dict, devices_dict, latency_dict, bandwidth_dict, num_stage
