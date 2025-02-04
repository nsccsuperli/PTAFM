source ./ip_list.sh

for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  #scp -P 222 /workspace/Hybrid_PTAFM/scripts/aws_run_gpt3_optimal_Ngpu_training.sh  root@"$ip":/workspace/Hybrid_PTAFM/scripts/
  scp -P 222 /workspace/Hybrid_PTAFM/pipeline_parallel/dist_gpipe_pipeline_async.py root@"$ip":/workspace/Hybrid_PTAFM/pipeline_parallel/
  #scp -P 222 /workspace/Hybrid_PTAFM/dist_runner.py root@"$ip":/workspace/Hybrid_PTAFM/
  #scp -P 222 /workspace/Hybrid_PTAFM/comm/comm_utils.py  root@"$ip":/workspace/Hybrid_PTAFM/comm/
  #scp -P 222 /workspace/Hybrid_PTAFM/utils/dist_train_utils.py root@"$ip":/workspace/Hybrid_PTAFM/utils/
  #scp -P 222 /workspace/Hybrid_PTAFM/scripts/local_scripts/gpt3_xl_pp8_dp8.sh root@"$ip":/workspace/Hybrid_PTAFM/scripts/local_scripts/
  #scp -P 222 /workspace/Hybrid_PTAFM/scripts/local_scripts/local_kill_process.sh root@"$ip":/workspace/Hybrid_PTAFM/scripts/local_scripts/
done
wait
