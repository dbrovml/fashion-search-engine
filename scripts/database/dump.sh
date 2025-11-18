set -a && source .env && set +a
docker exec -t $POSTGRES_CONTAINER_NAME pg_dump --no-owner --no-privileges -h localhost -U $POSTGRES_USER -d $POSTGRES_DB > data/local_dump.sql