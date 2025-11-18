set -a && source .env && set +a
psql $RENDER_DB_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql $RENDER_DB_URL -c "DROP SCHEMA IF EXISTS item CASCADE;"
psql $RENDER_DB_URL < data/local_dump.sql