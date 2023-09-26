if [ ! -f ./requirements_installed ]; 
then 
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    python -m ipykernel install --user --name myenv --display-name "Python(myenv)"
    touch requirements_installed
fi 

echo "dependências instaladas"
python main.py --gpu_ids=0,1,2,3 --lr=3.5e-4 --P=16 --K=12 --tau=0.04 --beta=0.999 --k1=30 --sampling=mean --lambda_hard=0.5 --num_iter=7 --momentum_on_feature_extraction=0 --target=Duke --path_to_save_models=models --path_to_save_metrics=metrics --version=version_name --eval_freq=5
