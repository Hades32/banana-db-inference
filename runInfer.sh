#!/bin/bash

# maybe make helper website from this

start=$(date)

callID=$(curl -s -XPOST 'https://api.banana.dev/start/v4/' -H 'Content-Type: application/json' -d '
{
    "startOnly": true,
    "apiKey": "'$API_KEY'",
    "modelKey": "'$MODEL_KEY'",
    "modelInputs": {
        "S3_ENDPOINT": "'$S3_ENDPOINT'",
        "S3_BUCKET": "'$S3_BUCKET'",
        "S3_KEY": "'$S3_KEY'",
        "S3_SECRET": "'$S3_SECRET'",
        "S3_REGION": "'$S3_REGION'",
        "id": "rita_6306431069847477",
        "prompts":  [
"a photo of sks person, glowing lights, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
"a photo of sks person with brown eyes, by magali villeneuve, artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, golden ratio, trending on art station ",
"portrait of sks person as a viking, full body concept, long hair braided, long beard braided, close portrait, fantasy, intricate, highly detailed, symmetry, dynamic lighting, attack on titan, artstation, digital painting, sharp focus, smooth, illustration, big muscles, art by argerm and greg rutkowski",
"portrait of a young ruggedly handsome but joyful sks person, male, masculine, strong, upper body, d & d, fantasy, seductive smirk, intricate, elegant, highly detailed, digital painting, artstation, concept art, matte, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
"hyperrealistic mixed media image of sks person, stunning 3 d render inspired art by greg rutkowski and xiang duan and thomas eakes, perfect facial symmetry, realistic, highly detailed attributes and atmosphere, dim volumetric cinematic lighting, 8 k octane extremely hyper - detailed render, post - processing, masterpiece",
"sks person in the style of Rick and Morty, unibrow, white robe, big eyes, realistic portrait, symmetrical, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, cinematic lighting, art by artgerm and greg rutkowski and alphonse mucha",
"plucky charming sks person rogue climbing ship ladder, naval background, fantasy, D&D 5e, 5th edition, portrait, piercing stare, highly detailed, digital painting, HD, artstation, concept art, matte, sharp focus, illustration, art by artgerm and greg rutkowski",
"Portrait of a middle aged sks person wearing a ceremonial uniform and a crown, male, detailed face, renaissance, highly detailed, cinematic lighting, digital art painting by greg rutkowski",
"concept art of sks person, cinematic shot, oil painting by jama jurabaev, extremely detailed, brush hard, artstation, high quality, brush stroke white background",
"masterpiece portrait of a clean shaved RPG sks person with a big nose and messy hair, D&D, fantasy, highly detailed, digital painting, sharp focus, illustration, art by artgerm and Edmiston and greg rutkowski and magali villeneuve",
"a photo of sks person, animation pixar style, by magali villeneuve, artgerm, jeremy lipkin and michael garmash, rob rey and kentaro miura style, golden ratio, trending on art station ",
"a photo of sks person, glowing lights, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
"a photo of sks person, dynamic lighting, cyber punk, color block, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
"portrait of sks person by greg rutkowski, young, attractive, highly detailed portrait, scifi, digital painting, artstation, concept art, smooth, sharp foccus ilustration, artstation hq, colorful background, saturated",
"portrait of sks person, painting in the style of Leonardo Da Vinci",
"portrait of sks person, painting in the style of Rembrant",
"portrait of sks person, painting in the style of Van Gogh",
"portrait of sks person in adaptation ( 2 0 0 2 ), highly detailed, centered, solid color background, digital painting, artstation, concept art, smooth, sharp focus, illustration, artgerm, donato giancola, joseph christian leyendecker, les edwards, ed repka, wlop, artgerm",
"a close-up portrait photo with intricte details of a screaming sks person in front of a burning bridge",
"a close-up portrait photo with intricte details of sks person diving in the red sea with lots of colorful fish in the background",
"a photo with intricte details of sks person in driving a go cart",
"a photo with intricte details of a roman bust of sks person in an art gallery"
        ],
        "height":  512,
        "width":  512,
        "num_inference_steps":  50,
        "guidance_scale":  8.0,
        "seed": 4242424242424223
    }
}' | jq -j .callID)

echo "started ${callID} at ${start}"

while export result=$(curl -s -XPOST 'https://api.banana.dev/check/v4/' -H 'Content-Type: application/json' -d'
{    
    "apiKey": "'$API_KEY'",    
    "callID": "'$callID'"
}') && echo "$result" | grep running > /dev/null ; do 
    echo -n .
done

echo "$result" | jq .
echo "$start until $(date)"
