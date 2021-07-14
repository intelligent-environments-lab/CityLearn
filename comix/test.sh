currdir=/home/users/bliu/research/CityLearn/comix

export CUDA_VISIBLE_DEVICES=0 && python src/main.py --config=comix-naf --env-config=citylearn with t_max=34400 seed=1 evaluate=True test_nepisode=10 runner=episode batch_size_run=1 batch_size=1 checkpoint_path=$currdir/results/models/comix-naf__2021-07-10_02-52-01/ save_replay=True > logs/comix-naf-test.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1 && python src/main.py --config=covdn --env-config=citylearn with t_max=34400 seed=1 evaluate=True test_nepisode=10 runner=episode batch_size_run=1 batch_size=1 checkpoint_path=$currdir/results/models/covdn__2021-07-10_02-52-01/ save_replay=True > logs/covdn-test.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2 && python src/main.py --config=facmaddpg --env-config=citylearn with t_max=34400 seed=1 evaluate=True test_nepisode=10 runner=episode batch_size_run=1 batch_size=1 checkpoint_path=$currdir/results/models/facmaddpg__2021-07-10_02-52-01/ save_replay=True > logs/facmaddpg-test.log 2>&1 &

#export CUDA_VISIBLE_DEVICES=1 && python src/main.py --config=maddpg    --env-config=citylearn with t_max=210242 seed=1 > logs/1-maddpg.log    2>&1 &
#export CUDA_VISIBLE_DEVICES=2 && python src/main.py --config=iql-cem   --env-config=citylearn with t_max=210242 seed=1 > logs/1-iql-cem.log   2>&1 &
#export CUDA_VISIBLE_DEVICES=3 && python src/main.py --config=iql-naf   --env-config=citylearn with t_max=1051210 seed=1 > logs/1-iql-naf.log   2>&1 &
#export CUDA_VISIBLE_DEVICES=1 && python src/main.py --config=comix     --env-config=citylearn with t_max=100000 seed=1 > logs/1-comix.log     2>&1 &
#export CUDA_VISIBLE_DEVICES=2 && python src/main.py --config=comix-naf --env-config=citylearn with t_max=100000 seed=1 > logs/1-comix-naf.log 2>&1 &
#export CUDA_VISIBLE_DEVICES=3 && python src/main.py --config=covdn     --env-config=citylearn with t_max=100000 seed=1 > logs/1-covdn.log     2>&1 &
#export CUDA_VISIBLE_DEVICES=7 && python src/main.py --config=covdn-naf --env-config=citylearn with t_max=100000 seed=1 > logs/1-covdn-naf.log 2>&1 &
