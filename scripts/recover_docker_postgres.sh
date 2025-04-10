sudo cp temp_pg/postgresql.conf ./pgdata/postgresql.conf
sudo cp temp_pg/postgresql.auto.conf ./pgdata/postgresql.auto.conf
sudo cp temp_pg/pg_hba.conf ./pgdata/pg_hba.conf

sudo docker restart tpcc_workload
sleep 2