sudo cp temp_pg/postgresql.conf ./pgdata/postgresql.conf
sudo cp temp_pg/postgresql.auto.conf ./pgdata/postgresql.auto.conf
sudo cp temp_pg/pg_hba.conf ./pgdata/pg_hba.conf

echo 251314 | sudo -S docker restart benchbase
sleep 2