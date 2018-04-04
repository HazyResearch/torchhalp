parallel --jobs 1 "python main.py --opt SVRG --lr {1} --T {2} --num_epochs 10" ::: 0.001 0.005 0.05 0.025  ::: 782 1173 1564 1955 
