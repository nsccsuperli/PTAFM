# Update this list of priviate IPs for each instance before use.
private_ips=(
"192.168.0.5"
"192.168.0.4"
)


for ip in "${private_ips[@]}"
do
  ping -c 5 $ip
done
