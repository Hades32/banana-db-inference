#!/bin/bash

# maybe make helper website from this

start=$(date)

callID=$(curl -s -XPOST 'https://api.banana.dev/start/v4/' -H 'Content-Type: application/json' -d '
{
    "startOnly": true,
    "apiKey": "'$API_KEY'",
    "modelKey": "'$MODEL_KEY'",
    "modelInputs": {
        "id": "input-1",
        "prompt":  "a picture of a sks person",
        "height":  512,
        "width":  512,
        "num_inference_steps":  50,
        "guidance_scale":  8.5,
        "seed": 42
    }
}' | jq -j .callID)

echo "started ${callID} at ${start}"

while curl -s -XPOST 'https://api.banana.dev/check/v4/' -H 'Content-Type: application/json' -d'
{    
    "apiKey": "'$API_KEY'",    
    "callID": "'$callID'"
}' | grep running > /dev/null ; do 
    echo -n .
done

echo
curl -s -XPOST 'https://api.banana.dev/check/v4/' -H 'Content-Type: application/json' -d'
{    
    "apiKey": "'$API_KEY'",    
    "callID": "'$callID'"
}' | jq .
echo "$start until $(date)"
