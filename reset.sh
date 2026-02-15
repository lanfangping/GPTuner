echo "Drop database volumes"
sudo rm -rf pgdata
sleep 1
echo "Stop DB container"
bash stop.sh
sleep 2
echo "Start new DB container"
bash start.sh
sleep 2