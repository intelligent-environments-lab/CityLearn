currdir=$(dirname "$0")

#names=("comix" "comix-naf" "covdn" "covdn-naf" "facmaddpg" "maddpg")
#names=("comix" "comix-naf" "facmaddpg")
names=("comix")
gpus=(1 2 3)

for i in "${!names[@]}"
do
    #export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=8760 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 rnn_hidden_dim=256 naf_hidden_dim=256 checkpoint_path=$currdir/results/models/${names[i]}__1/ save_replay=True > logs/${names[i]}__1-test.log 2>&1 &
    #export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=8760 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 rnn_hidden_dim=256 naf_hidden_dim=256 checkpoint_path=$currdir/results/models/${names[i]}_rewardexp_1/ save_replay=True env_args.reward_style=exp > logs/${names[i]}_exp_1-test.log 2>&1 &
    export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=8760 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 rnn_hidden_dim=256 naf_hidden_dim=256 checkpoint_path=$currdir/results/models/${names[i]}_rewardramping_abs_1/ save_replay=True env_args.reward_style=ramping_abs > logs/${names[i]}_rampingabs_1-test.log 2>&1 &
    #export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=8760 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 rnn_hidden_dim=256 naf_hidden_dim=256 checkpoint_path=$currdir/results/models/${names[i]}_rewardramping_square_1/ save_replay=True env_args.reward_style=ramping_square > logs/${names[i]}_rampingsq_1-test.log 2>&1 &
done
#export CUDA_VISIBLE_DEVICES=0 && python src/main.py --config=facmaddpg --env-config=citylearn with batch_size=512 rnn_hidden_dim=256 naf_hidden_dim=256 weight_decay=True weight_decay_factor=0.005 buffer_size=10000 test_interval=8759 target_update_tau=0.005 test_nepisode=1 optimizer_epsilon=0.0001 t_max=$tmax seed=$seed start_steps=7500 grad_norm_clip=10000 n_train=2 > logs/$seed-facmaddpg.log 2>&1 &
