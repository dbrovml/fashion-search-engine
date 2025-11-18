set -a && source .env && set +a
ngrok config add-authtoken $NGROK_AUTH_TOKEN
ngrok http 8000