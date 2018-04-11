parallel --jobs 3 "python main.py --num_epochs 150 --lr {1} --opt HALP --net ResNet --T {3} --mu {2}" ::: 0.1 ::: 20 5 ::: 165 391 1173 782 
