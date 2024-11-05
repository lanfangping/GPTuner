# sudo rm /var/lib/postgresql/14/main/postgresql.auto.conf
# sudo docker exec -u root pg_benchmark rm /var/lib/postgresql/data/postgresql.auto.conf
# sleep 2
# su - postgres -c '/usr/lib/postgresql/14/bin/pg_ctl restart -D /var/lib/postgresql/14/main/ -o "-c config_file=/etc/postgresql/14/main/postgresql.conf"'

sudo cp temp_pg/postgresql.conf ./pgdata/postgresql.conf
sudo cp temp_pg/postgresql.auto.conf ./pgdata/postgresql.auto.conf
sudo cp temp_pg/pg_hba.conf ./pgdata/pg_hba.conf

sudo docker restart test_pg_benchmark
sleep 2