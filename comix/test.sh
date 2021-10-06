#names=("comix" "comix-naf" "covdn" "covdn-naf" "facmaddpg" "maddpg")
#names=("covdn-naf" "iql-cem" "maddpg")
names=("iql-cem")
gpus=(1 2 3)

for i in "${!names[@]}"
do
    #reward=ramping_square
    #export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=8760 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 rnn_hidden_dim=256 naf_hidden_dim=256 checkpoint_path=./results/models/${names[i]}_reward$reward\_1/ save_replay=True env_args.reward_style=$reward > logs/1-$reward-${names[i]}-test.log 2>&1 &
    reward=exp
    export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=8760 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 rnn_hidden_dim=256 naf_hidden_dim=256 checkpoint_path=./results/models/${names[i]}_reward$reward\_1/ save_replay=True env_args.reward_style=$reward > logs/1-$reward-${names[i]}-test.log 2>&1 &
    #reward=marlisa
    #export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=8760 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 rnn_hidden_dim=256 naf_hidden_dim=256 checkpoint_path=./results/models/${names[i]}_reward$reward\_1/ save_replay=True env_args.reward_style=$reward > logs/1-$reward-${names[i]}-test.log 2>&1 &
    #reward=mixed
    #export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=8760 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 rnn_hidden_dim=256 naf_hidden_dim=256 checkpoint_path=./results/models/${names[i]}_reward$reward\_1/ save_replay=True env_args.reward_style=$reward > logs/1-$reward-${names[i]}-test.log 2>&1 &
done
