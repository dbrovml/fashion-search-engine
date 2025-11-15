#!/bin/bash
set -e
source .env

ssh-keyscan -H "$LAMBDA_HOST" >> ~/.ssh/known_hosts
SSH="ssh -i $LAMBDA_SSH_KEY $LAMBDA_USER@$LAMBDA_HOST"
$SSH -f -N -R 15432:localhost:5432
rsync -avzR -e "ssh -i $LAMBDA_SSH_KEY" src/ pyproject.toml .env $LAMBDA_USER@$LAMBDA_HOST:$LAMBDA_DIR/
$SSH "
cd $LAMBDA_DIR \
&& export PATH=\$HOME/.local/bin:\$PATH \
&& python3 -m pip install uv \
&& uv sync \
&& export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
&& export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
&& aws s3 sync s3://$AWS_S3_BUCKET/images/ data/images/ \
&& ENVIRONMENT=remote uv run python -m src.embedding.items \
&& ENVIRONMENT=remote uv run python -m src.embedding.colors
"