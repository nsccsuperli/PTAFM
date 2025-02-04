sudo tc qdisc add dev eth0 root handle 1: htb default 1000
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 200000.0mbps
sudo tc class add dev eth0 parent 1: classid 1:2 htb rate 1.0Gbps
sudo tc qdisc add dev eth0 parent 1:1 handle 10: netem delay 0.1ms
sudo tc qdisc add dev eth0 parent 1:2 handle 20: netem delay 100.0ms
sudo tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dst 172.16.0.34/32 flowid 1:1
sudo tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dst 172.16.0.35/32 flowid 1:2
