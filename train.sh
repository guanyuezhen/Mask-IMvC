#!/bin/bash

lr="1e-3"
train_epochs=1500

data_names=(
"cub-2view"
)

miss_rates=(0.0 0.1 0.3 0.5 0.7 0.9) # differernt miss ratios
alphas=(-3 -2 -1 0 1 2 3) # differernt hyperparameters

for data_name in "${data_names[@]}"; do
  for miss_rate in "${miss_rates[@]}"; do
      for alpha in "${alphas[@]}"; do
          python ./main.py --data_name "$data_name" --lr "$lr" --train_epochs "$train_epochs" --miss_rate "$miss_rate" --alpha "$alpha"
      done
    done
done
