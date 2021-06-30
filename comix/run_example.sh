export CUDA_VISIBLE_DEVICES=0 && mkdir -p logs && python src/main.py --config=comix --env-config=citylearn with t_max=105120 > logs/comix.log
