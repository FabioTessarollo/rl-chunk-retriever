#!/bin/bash

uv run 6_train.py \
    --proj_dim 512 \
    --gamma 0.99 \
    --epsilon_min 0.01 \
    --epsilon_decay 0.99993 \
    --batch_size 64 \
    --replay_capacity 10000 \
    --lr 0.0005 \
    --target_update 100 \
    --epochs 50 \
    --max_exp_loops 3 \
    --action_dim 4 \
    --dropout_p 0 \
    --scheduler_type plateau \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001

uv run 6_train.py \
    --proj_dim 256 \
    --gamma 0.99 \
    --epsilon_min 0.01 \
    --epsilon_decay 0.99993 \
    --batch_size 64 \
    --replay_capacity 10000 \
    --lr 0.0005 \
    --target_update 20 \
    --epochs 50 \
    --max_exp_loops 3 \
    --action_dim 4 \
    --dropout_p 0 \
    --scheduler_type plateau \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001

uv run 6_train.py \
    --proj_dim 256 \
    --gamma 0.99 \
    --epsilon_min 0.01 \
    --epsilon_decay 0.99993 \
    --batch_size 32 \
    --replay_capacity 10000 \
    --lr 0.00025 \
    --target_update 100 \
    --epochs 50 \
    --max_exp_loops 3 \
    --action_dim 4 \
    --dropout_p 0 \
    --scheduler_type plateau \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001

uv run 6_train.py \
    --proj_dim 256 \
    --gamma 0.99 \
    --epsilon_min 0.01 \
    --epsilon_decay 0.99993 \
    --batch_size 128 \
    --replay_capacity 10000 \
    --lr 0.001 \
    --target_update 100 \
    --epochs 50 \
    --max_exp_loops 3 \
    --action_dim 4 \
    --dropout_p 0 \
    --scheduler_type plateau \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001

uv run 6_train.py \
    --proj_dim 512 \
    --gamma 0.99 \
    --epsilon_min 0.01 \
    --epsilon_decay 0.99997 \
    --batch_size 64 \
    --replay_capacity 10000 \
    --lr 0.0005 \
    --target_update 80 \
    --epochs 50 \
    --max_exp_loops 3 \
    --action_dim 4 \
    --dropout_p 0.1 \
    --scheduler_type plateau \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001

uv run 6_train.py \
    --proj_dim 512 \
    --gamma 0.99 \
    --epsilon_min 0.1 \
    --epsilon_decay 0.99999 \
    --batch_size 128 \
    --replay_capacity 20000 \
    --lr 0.0005 \
    --target_update 250 \
    --epochs 100 \
    --max_exp_loops 3 \
    --action_dim 4 \
    --dropout_p 0.1 \
    --scheduler_type plateau \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001

uv run 6_train.py \
    --proj_dim 256 \
    --gamma 0.95 \
    --epsilon_min 0.01 \
    --epsilon_decay 0.99997 \
    --batch_size 64 \
    --replay_capacity 10000 \
    --lr 0.0005 \
    --target_update 100 \
    --epochs 50 \
    --max_exp_loops 1 \
    --action_dim 3 \
    --dropout_p 0 \
    --scheduler_type plateau \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001

uv run 6_train.py \
    --proj_dim 256 \
    --gamma 0.95 \
    --epsilon_min 0.01 \
    --epsilon_decay 0.99997 \
    --batch_size 64 \
    --replay_capacity 10000 \
    --lr 0.0005 \
    --target_update 50  \
    --epochs 50 \
    --max_exp_loops 1 \
    --action_dim 3 \
    --dropout_p 0 \
    --scheduler_type step \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001
