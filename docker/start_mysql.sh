#!/usr/bin/env bash
set -euo pipefail

if docker compose version >/dev/null 2>&1; then
  docker compose -f docker/docker-compose.mysql.yml up -d
else
  docker-compose -f docker/docker-compose.mysql.yml up -d
fi
