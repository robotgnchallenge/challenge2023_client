terminator --new-tab -e "docker run --rm --gpus 'all,\"capabilities=compute,utility\"' -it  --net host task_client /bin/bash -c \"source /root/anaconda3/etc/profile.d/conda.sh; conda deactivate; conda activate robotmani; cd /root/client_ws; source devel/setup.bash; roscore\";
exec bash" &
sleep 5

terminator --new-tab -e "
docker exec -it $(docker container ls  | grep 'task_client' | awk '{print $1}') /bin/bash -c \"source /root/anaconda3/etc/profile.d/conda.sh; conda deactivate; conda activate robotmani; cd /root/client_ws; source devel/setup.bash; cd src/task_client/demo/src; python object_search_planner_socket.py\";
exec bash" &
sleep 1

terminator --new-tab -e "
docker exec -it $(docker container ls  | grep 'task_client' | awk '{print $1}') /bin/bash -c \"source /root/anaconda3/etc/profile.d/conda.sh; conda deactivate; conda activate robotmani; cd /root/client_ws; source devel/setup.bash; rosrun task_client construct_semmap_with_socket\";
exec bash" &
sleep 1

terminator --new-tab -e "
docker exec -it $(docker container ls  | grep 'task_client' | awk '{print $1}') /bin/bash -c \"source /root/anaconda3/etc/profile.d/conda.sh; conda deactivate; conda activate robotmani; cd /root/client_ws; source devel/setup.bash; cd src/task_client/demo/src; python task_client_demo.py\";
killall -e roslaunch construct_semma python roscore;
exec bash" &
sleep 1

