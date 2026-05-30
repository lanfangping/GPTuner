#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${MYSQL_DOCKER_COMPOSE_FILE:-docker/docker-compose.mysql.yml}"
SERVICE_NAME="${MYSQL_DOCKER_SERVICE:-mysql}"
CONTAINER_NAME="${MYSQL_DOCKER_CONTAINER:-mysql-8}"
MYSQL_RECOVERY_USER="${MYSQL_RECOVERY_USER:-root}"
MYSQL_RECOVERY_PASSWORD="${MYSQL_RECOVERY_PASSWORD:-${MYSQL_ROOT_PASSWORD:-mysql}}"

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "Docker compose file not found: ${COMPOSE_FILE}" >&2
  exit 1
fi

COMPOSE_DIR="$(cd "$(dirname "${COMPOSE_FILE}")" && pwd)"
MYSQL_DATA_DIR="${MYSQL_DOCKER_DATA_DIR:-${COMPOSE_DIR}/mysql-8-data}"
MYSQL_PERSISTED_CONFIG="${MYSQL_DATA_DIR}/mysqld-auto.cnf"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -f "${COMPOSE_FILE}" "$@"
  else
    docker-compose -f "${COMPOSE_FILE}" "$@"
  fi
}

mysql_exec() {
  docker exec -e MYSQL_PWD="${MYSQL_RECOVERY_PASSWORD}" "${CONTAINER_NAME}" \
    mysql -u"${MYSQL_RECOVERY_USER}" \
    -e "$1"
}

wait_for_mysql() {
  echo "Wait for MySQL to accept connections"
  for _ in $(seq 1 30); do
    if docker exec -e MYSQL_PWD="${MYSQL_RECOVERY_PASSWORD}" "${CONTAINER_NAME}" \
      mysqladmin ping -u"${MYSQL_RECOVERY_USER}" --silent >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "MySQL did not become ready in time" >&2
  return 1
}

backup_persisted_config() {
  if [[ ! -f "${MYSQL_PERSISTED_CONFIG}" ]]; then
    echo "No persisted MySQL config found at ${MYSQL_PERSISTED_CONFIG}"
    return 0
  fi

  local backup_path
  backup_path="${MYSQL_PERSISTED_CONFIG}.bak.$(date +%Y%m%d%H%M%S)"

  echo "Back up and remove persisted MySQL config: ${MYSQL_PERSISTED_CONFIG}"
  mv "${MYSQL_PERSISTED_CONFIG}" "${backup_path}"
}

echo "Force-stop MySQL container"
docker kill "${CONTAINER_NAME}" >/dev/null 2>&1 || true
sleep 5

backup_persisted_config

echo "Start MySQL service from ${COMPOSE_FILE}"
compose up -d "${SERVICE_NAME}" >/dev/null
wait_for_mysql

echo "Reset MySQL persisted variables"
mysql_exec "RESET PERSIST;"

echo "Reset MySQL optimizer cost tables"
mysql_exec "UPDATE mysql.server_cost SET cost_value = NULL; UPDATE mysql.engine_cost SET cost_value = NULL; FLUSH OPTIMIZER_COSTS;"

echo "Restart MySQL service to reload defaults"
compose restart "${SERVICE_NAME}" >/dev/null
wait_for_mysql

echo "MySQL recovery complete"
