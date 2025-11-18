set -a && source .env && set +a
docker exec -it $POSTGRES_CONTAINER_NAME psql -U $POSTGRES_USER -d $POSTGRES_DB