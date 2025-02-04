# bash aws_run_gpt3_optimal_Ngpu_training.sh gpt3_xl_pp8_dp8.sh 1 2 3 256 1
#                                            1     			 		  	2	3	4	5	 6	

source ./ip_list.sh
echo -en "\033[37;40;7m ---START PAFM--- ./script/aws_run_gpt3_optimal_Ngpu_training.sh  \033[0m \n"

# Change the script here for different settings.
############################################################
# world_size=64
world_size=48
############################################################

script=$1
gpu_count_mode=$2 # 1
# 根据IPlist对rank进行映射
#rank_map=[12, 13, 6, 2, 3, 0, 1, 7, 8, 9, 10, 11, 4, 5, 18, 19, 20, 21, 22, 23, 42, 43, 36, 37, 38, 39, 40, 41, 26, 27, 28, 29, 34, 35, 44, 45, 46, 47, 24, 25, 30, 31, 32, 33, 14, 15, 16,17]
rank_map=(0 1 6 2 3 12 13 7 8 9 10 11 4 5 18 19 20 21 22 23 42 43 36 37 38 39 40 41 26 27 28 29 34 35 44 45 46 47 24 25 30 31 32 33 14 15 16 17)
#rank_map=(0 1 39 40 41 29 30 18 19 20 21 22 37 38 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 23 24 25 26 27 28 31 32 33 34 35 36 42 43 44 45 46 47)
#rank_mapping_id=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 24 30 36 42 48 49 50 51 52 53 54 60 66 67 68 69 70 71 72 78 84 90 96 102 103 104 105 106 107)
if [ $gpu_count_mode -eq 1 ]
then
# Change the script here for different settings.
############################################################
  # nodes_per_node=(8 8 8 8 8 8 8 8)
  # 2,1, 2,2,5, 2 2 2 2 2 6 2 2 2 1 1 1 1 2 2 2 4
  #nodes_per_node=(2 1)
  nodes_per_node=(2 1 2 2 5 2 2 2 2 2 6 2 2 2 1 1 1 1 2 2 2 4)
############################################################
else
  echo "Error! Not valid arguments."
  exit 1
fi




ga_step=$3    # 2
num_layers=$4 # 3
batch_size=$5 # 256

log_mode='optimal_map'
echo "\$#: "$#


declare -i rank=0
for node_rank in "${!ips[@]}"
do
  # 红色
  echo -en "\033[37;41;1m Issue command $script in Rank-${node_rank} node: ${ips[node_rank]} \033[0m \n"
  if [ $# -eq 6 ]
  then
    case=$6
    
    echo -e "Running in heterogeneous network: Case-$case & num of GPU is ${nodes_per_node[$node_rank]}"
    
    for (( i=0; i < ${nodes_per_node[$node_rank]}; i++))
    do
      echo "Running in device $node_rank & cuda $i with ip ${ips[node_rank]} rankid- ${rank_map[rank]}"
      ## ssh
      #rankids = ${rank_map[rank]}
      ssh  root@"${ips[node_rank]}" -p $port "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "${rank_map[rank]}" "$i" "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$case" &
#ssh  root@"${ips[node_rank]}" -p $port "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" "$i" "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$case" & 
      rank+=1
    done
  else
    echo "Error! Not valid arguments."
    exit 1
  fi
  # exit

done
echo -en "\033[37;40;7m ---END--- ./script/aws_run_gpt3_optimal_Ngpu_training.sh \033[0m \n"

wait
