#!/usr/bin/env bash
set -euo pipefail

docker exec -it mysql-8 mysql -uroot -pmysql workload
