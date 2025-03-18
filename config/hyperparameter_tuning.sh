#!/bin/bash

config_file="crohme.yaml"

new_train_batch_size=$1

if [ -z "$new_train_batch_size" ]; then
  echo "Error: Please provide a new train_batch_size value."
  exit 1
fi

echo $new_train_batch_size

sed -n -i '' "s/train_batch_size: [0-9][0-9]/train_batch_size: $new_train_batch_size/g" $config_file


