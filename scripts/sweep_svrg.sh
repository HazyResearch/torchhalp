parallel --jobs 2 "python main.py --opt SVRG --lr {1} --T {2} --num_epochs 10" ::: 0.005 0.001 0.01 0.025 ::: 782 1173 1564 1955 
