#!/bin/bash
set -e
source .env

ssh-keyscan -H "$LAMBDA_HOST" >> ~/.ssh/known_hosts
SSH="ssh -i $LAMBDA_SSH_KEY $LAMBDA_USER@$LAMBDA_HOST"
$SSH -f -N -R 15432:localhost:5432
rsync -avzR --mkpath -e "ssh -i $LAMBDA_SSH_KEY" src/ pyproject.toml .env $LAMBDA_USER@$LAMBDA_HOST:$LAMBDA_DIR/
rsync -avzR --mkpath -e "ssh -i $LAMBDA_SSH_KEY" data/images/ $LAMBDA_USER@$LAMBDA_HOST:$LAMBDA_DIR/
$SSH "
cd $LAMBDA_DIR \
&& export PATH=\$HOME/.local/bin:\$PATH \
&& python3 -m pip install uv \
&& uv sync \
&& ENVIRONMENT=remote uv run python -m src.embedding.items \
&& ENVIRONMENT=remote uv run python -m src.embedding.colors
"