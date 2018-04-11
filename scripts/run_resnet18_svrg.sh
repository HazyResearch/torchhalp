parallel --jobs 3 "python main.py --num_epochs 150 --lr {1} --opt SVRG --net ResNet --T {2}" ::: 0.1 0.05 ::: 165 391 782 1173
