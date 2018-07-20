#!/bin/bash
fileid="1i19e16WKDJ_iTW7ecgK_Ya5g43sClMby"
filename="english.bin.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

curl -O https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec
