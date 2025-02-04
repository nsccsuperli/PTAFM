source /workspace/DT-FM/scripts/ip_list.sh

docker_1="418"
docker_2="f92"


# ssh -i $pem_file root@$master_ip  "docker cp $docker_1:/workspace/DT-FM/ /tmp"
# ssh -i $pem_file root@$master_ip  "scp -r /tmp/DT-FM root@192.168.0.4:/tmp/DT-FM"
# ssh -i $pem_file root@192.168.0.4  "docker cp /tmp/DT-FM $docker_2:/workspace/"

scp -i  $pem_file -r -P 9022 /workspace/DT-FM root@192.168.0.4:/workspace

clear
pkill -9 python
ssh -i $pem_file  -p 9022  root@192.168.0.4 "pkill -9 python"
