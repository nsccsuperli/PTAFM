# python generate_heterogeneous_tc.py --case 3/4/5 --rank $rank --nodes ips的数量


import argparse
import numpy as np
from generate_sim_com_matrices import *

# No space for IPs!!!!
private_ip = [
    "192.168.1.6",
    "192.168.1.25",
    "192.168.1.5",
    "192.168.1.17",
    "192.168.1.18",
    "192.168.1.19",
    "192.168.1.26",
    "192.168.1.27",
    "192.168.1.28",
    "192.168.1.29",
    "192.168.1.7",
    "192.168.1.8",
    "192.168.1.9",
    "192.168.1.11",
    "192.168.1.12",
    "192.168.1.13",
    "192.168.1.14"
]


def get_delay_bandwidth(args):
#    if args.case == '1':
#        return simulate_1_datacenter(args.nodes)
#    elif args.case == '2':
#        return simulate_2_datacenter_spot_gpu(args.nodes)
#   elif args.case == '3':
#        return simulate_3_multi_universities(args.nodes)
#    elif args.case == '4':
#        return simulate_4_regional_geo_distributed(args.nodes)
#    elif args.case == '5':
#        return simulate_5_worldwide_geo_distributed(args.nodes)
#    else:
#        assert False

    region_ip = {}
    region_ip[6] = ["192.168.1.6"]
    region_ip[2] = ["192.168.1.25", "192.168.1.5"]
    region_ip[3] = ["192.168.1.17", "192.168.1.18", "192.168.1.19"]
    region_ip[4] = ["192.168.1.26", "192.168.1.27", "192.168.1.28", "192.168.1.29"]
    region_ip[5] = ["192.168.1.7", "192.168.1.8", "192.168.1.9"]
    region_ip[1] = ["192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14"]
    region_delay = np.array([[0.1, 167, 184, 44, 124, 200],
                             [167, 0.1, 58, 62, 86, 35],
                             [184, 58, 0.1, 103, 126, 78],
                             [44, 62, 103, 0.1, 41, 154],
                             [124, 86, 126, 41, 0.1, 80],
                             [200, 35, 78, 154, 80, 0.1]])

    region_bandwidth = np.array([[200, 0.332, 0.956, 0.634, 1.119, 1.280],
                                 [0.332, 200, 1.189, 0.388, 0.308, 0.853],
                                 [0.956, 1.189, 200, 0.747, 0.617, 0.982],
                                 [0.634, 0.388, 0.747, 200, 1.190, 1.129],
                                 [1.119, 0.308, 0.617, 1.190, 200, 0.485],
                                 [1.280, 0.853, 0.982, 1.129, 0.485, 200]])
    delay = np.zeros((17, 17))
    bandwidth = np.zeros((17, 17))

    for i in range(17):
        for j in range(17):
            for n in range(6):
                if private_ip[i] in region_ip[n + 1]:
                    i_index = n
            for n in range(6):
                if private_ip[j] in region_ip[n + 1]:
                    j_index = n
            delay[i, j] = region_delay[i_index, j_index]
            bandwidth[i, j] = region_bandwidth[i_index, j_index]
    _ = 0
    return delay, bandwidth, _


def generate_tc_scripts(args):
    assert args.nodes == len(private_ip)
    delay, bandwidth, _ = get_delay_bandwidth(args)
    with open("../scripts/tc_scripts/heterogeneous_setup_case"+str(args.case)+".sh", 'w') as script:
        tc_setting_dict = {}
        handle_i = 1
        for i in range(len(private_ip)):
            if i != args.rank:
                current_key = (delay[args.rank][i], bandwidth[args.rank][i])
                if current_key not in tc_setting_dict:
                    tc_setting_dict[current_key] = handle_i
                    handle_i += 1
        assert len(tc_setting_dict) <= 16
        # setup delay and bandwidth subclass qdisc
        script.write("sudo tc qdisc add dev eth0 root handle 1: prio bands {}\n"
                     .format(max(3, len(tc_setting_dict))))
        for key in tc_setting_dict.keys():
            current_delay, current_bandwidth = key
            handle_index = tc_setting_dict[key]
            limit_pkts = current_delay * 22500 * current_bandwidth
            script.write("sudo tc qdisc add dev eth0 parent 1:{} handle {}: netem delay {}ms rate {}Gbit limit {}\n"
                         .format(handle_index, handle_index*10, current_delay, current_bandwidth, limit_pkts))
        # setup filter
        for i in range(len(private_ip)):
            if i != args.rank:
                current_key = (delay[args.rank][i], bandwidth[args.rank][i])
                script.write("sudo tc filter add dev eth0 parent 1:0 protocol ip prio 1 u32 match ip dst {}/32 flowid 1:{}\n"
                             .format(private_ip[i], tc_setting_dict[current_key]))


def main():
    parser = argparse.ArgumentParser(description='Test PyTorch Distributed')
    parser.add_argument('--case', type=str, default='0', metavar='R', help='which case to generate.')
    parser.add_argument('--rank', type=int, default=0, metavar='R', help='rank for this IP')
    parser.add_argument('--nodes', type=int, default=17, metavar='R', help='Total number of nodes')
    args = parser.parse_args()
    generate_tc_scripts(args)


if __name__ == '__main__':
    main()
