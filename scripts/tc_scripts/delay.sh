DELAY_MS=$1
echo "TC set up delay to $1 ms."
sudo tc qdisc add dev eth0 root netem delay ${DELAY_MS}ms