parallel --jobs 2 "python main.py --num_epochs 10 --lr {1} --opt SVRG --net ResNet --T {2}" ::: 0.1 0.5 1.0 0.05 ::: 165 391 1173 782
