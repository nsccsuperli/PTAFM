sudo tc qdisc add dev eth0 root handle 1: prio bands 3
sudo tc qdisc add dev eth0 parent 1:1 handle 10: netem delay 11.0ms rate 1.12Gbit limit 277200.0
sudo tc filter add dev eth0 parent 1:0 protocol ip prio 1 u32 match ip dst 192.168.0.5/32 flowid 1:1
