source ./ip_list.sh

for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  ssh -p 222 root@"$ip" "bash -s" < ./tc_scripts/clear.sh &
done
wait
