#!/usr/bin/env bash
set -euo pipefail

echo "Stop MySQL container"
bash docker/stop_mysql.sh

echo "Drop MySQL database volume"
rm -rf docker/mysql-8-data

echo "Start new MySQL container"
bash docker/start_mysql.sh
