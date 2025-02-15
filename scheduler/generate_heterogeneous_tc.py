# python generate_heterogeneous_tc.py --case 3/4/5 --rank $rank --nodes ips的数量


import argparse
from generate_sim_com_matrices import *

# No space for IPs!!!!
private_ip = [
    "192.168.0.5", # node1
    "192.168.0.4"  # node2
]


def get_delay_bandwidth(args):
    if args.case == '1':
        return simulate_1_datacenter(args.nodes)
    elif args.case == '2':
        return simulate_2_datacenter_spot_gpu(args.nodes)
    elif args.case == '3':
        return simulate_3_multi_universities(args.nodes)
    elif args.case == '4':
        return simulate_4_regional_geo_distributed(args.nodes)
    elif args.case == '5':
        return simulate_5_worldwide_geo_distributed(args.nodes)
    else:
        assert False


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
        script.write("tc qdisc add dev ens3 root handle 1: prio bands {}\n"
                     .format(max(3, len(tc_setting_dict))))
        for key in tc_setting_dict.keys():
            current_delay, current_bandwidth = key
            handle_index = tc_setting_dict[key]
            limit_pkts = current_delay * 22500 * current_bandwidth
            script.write("tc qdisc add dev ens3 parent 1:{} handle {}: netem delay {}ms rate {}Gbit limit {}\n"
                         .format(handle_index, handle_index*10, current_delay, current_bandwidth, limit_pkts))
        # setup filter
        for i in range(len(private_ip)):
            if i != args.rank:
                current_key = (delay[args.rank][i], bandwidth[args.rank][i])
                script.write("tc filter add dev ens3 parent 1:0 protocol ip prio 1 u32 match ip dst {}/32 flowid 1:{}\n"
                             .format(private_ip[i], tc_setting_dict[current_key]))


def main():
    parser = argparse.ArgumentParser(description='Test PyTorch Distributed')
    parser.add_argument('--case', type=str, default='0', metavar='R', help='which case to generate.')
    parser.add_argument('--rank', type=int, default=0, metavar='R', help='rank for this IP')
    parser.add_argument('--nodes', type=int, default=64, metavar='R', help='Total number of nodes')
    args = parser.parse_args()
    generate_tc_scripts(args)


if __name__ == '__main__':
    main()
