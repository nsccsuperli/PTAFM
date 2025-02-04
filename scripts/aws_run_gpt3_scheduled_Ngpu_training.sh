source ./ip_list.sh


world_size=3

script=$1
gpu_count_mode=$2

if [ $gpu_count_mode -eq 1 ]
then
  nodes_per_node=(8 8 8 8 8 8 8 8)
elif [ $gpu_count_mode -eq 2 ]
then
  nodes_per_node=(2 1)
else
  echo "Error! Not valid arguments."
  exit 1
fi

# Random seed 2022
# rank_map=(0 2 32 33 4 10 7 45 36 8 51 26 11 5 53 1 40 23 37 14 13 43 54 21 57 35 63 18 6 24 16 22 38 3 58 61 44 27 52 30 15 9 39 47 48 41 31 20 12 28 34 42 17 55 19 25 56 60 59 50 49 46 29 62)
# Random seed 2023
rank_map=(0 1 2)
# Random seed 2024
# rank_map=(0 35 24 52 16 49 41 4 18 7 45 13 22 60 14 15 51 25 17 8 47 55 19 63 21 57 44 26 5 58 20 50 30 6 54 43 23 34 46 27 39 10 40 62 29 12 32 53 48 31 38 61 3 11 36 2 42 56 37 28 1 33 59 9)

ga_step=$3
num_layers=$4
batch_size=$5

log_mode=$6

declare -i rank_index=0

for node_rank in "${!ips[@]}"
do
  echo -en "\033[37;41;1m Issue command $script in Rank-${node_rank} node: ${ips[node_rank]} \033[0m \n"
  if [ $# -eq 7 ]
  then
    case=$7
    echo "Running in heterogeneous network: Case-$case"
    for (( i=0; i<${nodes_per_node[$node_rank]}; i++))
    do
      ssh -i $pem_file root@"${ips[node_rank]}" -p $port "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "${rank_map[rank_index]}" "$i" "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$case" &
      rank_index+=1
    done
  fi
done
wait