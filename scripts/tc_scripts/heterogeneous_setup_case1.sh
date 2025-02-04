sudo tc qdisc add dev eth0 root handle 1: prio bands 3
sudo tc qdisc add dev eth0 parent 1:1 handle 10: netem delay 0.0ms rate 3.125Gbit limit 0.0
sudo tc filter add dev eth0 parent 1:0 protocol ip prio 1 u32 match ip dst 192.168.0.4/32 flowid 1:1
