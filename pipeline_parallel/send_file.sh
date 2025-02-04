ips=(
#"192.168.1.11"
"172.16.0.35"
"192.168.0.5"
"192.168.0.4"
"192.168.0.2"
"172.16.0.34"
"192.168.1.12"
"192.168.1.13"
"192.168.1.14"
"192.168.1.25"
"192.168.1.5"
"192.168.1.17"
"192.168.1.18"
"192.168.1.19"
"192.168.1.26"
"192.168.1.27"
"192.168.1.28"
"192.168.1.29"
"192.168.1.7"
"192.168.1.8"
"192.168.1.9"
"192.168.1.6"
)

local_file_path="/workspace/PTAFM_V1/pipeline_parallel/dist_gpipe_pipeline_async.py"
remote_directory="/workspace/PTAFM_V1/pipeline_parallel"


# 遍历每个 IP 地址并使用 scp 发送文件
for ip in "${ips[@]}"
do
    echo "Sending file to $ip..."
    scp  -P 222 "$local_file_path" root@"$ip":"$remote_directory"
    if [ $? -eq 0 ]; then
        echo "File sent to $ip successfully."
    else
        echo "Failed to send file to $ip."
    fi
done

