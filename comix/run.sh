export CUDA_VISIBLE_DEVICES=0 && python src/main.py --config=comix-naf --env-config=citylearn with t_max=1051210 > logs/comix-naf.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1 && python src/main.py --config=comix     --env-config=citylearn with t_max=1051210 > logs/comix.log     2>&1 &
export CUDA_VISIBLE_DEVICES=2 && python src/main.py --config=covdn-naf --env-config=citylearn with t_max=1051210 > logs/covdn-naf.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3 && python src/main.py --config=covdn     --env-config=citylearn with t_max=1051210 > logs/covdn.log     2>&1 &
