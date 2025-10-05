
# and set reward in case of not skip of top 3 even if the chunk is rel
uv run 6_train.py \
    --proj_dim 512 \
    --gamma 0.99 \
    --epsilon_min 0.01 \
    --epsilon_decay 0.99995 \
    --batch_size 128 \
    --replay_capacity 10000 \
    --lr 0.00005 \
    --target_update 300 \
    --epochs 100 \
    --max_exp_loops 3 \
    --action_dim 4 \
    --dropout_p 0 \
    --scheduler_type cosine \
    --per_alpha 0.6 \
    --per_beta 0.4 \
    --per_beta_increment 0.001