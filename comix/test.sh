currdir=$(dirname "$0")

names=("comix", "covdn-naf" "iql-cem" "iql-naf" "maddpg")
gpus=(1 2 3 5 7)

for i in "${!names[@]}"
do
    export CUDA_VISIBLE_DEVICES=${gpus[i]} && python src/main.py --config=${names[i]} --env-config=citylearn with t_max=262800 seed=1 evaluate=True test_nepisode=10 runner=episode batch_size_run=1 batch_size=1 checkpoint_path=$currdir/results/models/${names[i]}__1/ save_replay=True > logs/${names[i]}__1-test.log 2>&1 &
done
