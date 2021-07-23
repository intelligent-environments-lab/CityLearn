currdir=$(dirname "$0")

names=("comix-naf" "covdn" "facmaddpg")
gpus=(1 2 3)

for i in "${!names[@]}"
do
    export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=35040 seed=1 evaluate=True test_nepisode=1 runner=episode batch_size_run=1 batch_size=1 checkpoint_path=$currdir/results/models/${names[i]}__1/ save_replay=True > logs/${names[i]}__1-test.log 2>&1 &
done
