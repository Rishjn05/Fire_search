$env:OPENBLAS_NUM_THREADS="1"
D:\fire_search_project\.venv\Scripts\Activate.ps1


D:\fire_search_project\.venv\Scripts\Activate.ps1
cd D:\Downloads\mappo_fire\mappo_fire
ray stop --force
$env:OPENBLAS_NUM_THREADS="1"
$env:PYTHONUNBUFFERED="1"
python train_mappo.py --iterations 200 2>&1 | Tee-Object -FilePath train_log_1.txt

python generate_test_split_gifs.py --agents 3
python generate_test_split_gifs.py --agents 4
python zero_shot_eval_marl.py --agents 4