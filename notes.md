VERSION 1:
    - 9020/10000 after one epoch of training
    - nevermind i was applying softmax in the wrong place, it was only guessing zeroes
VERSION 1.5:
    - after fix, 79.39% accuracy after 1 epoch - 7939/10000 -- learning rate of .001
    - after 5 epochs: 
    - downside: still 2 min - 2 min 30s per epoch -- improved to 1 min 36 s per epoch
    - 87.57% accuracy after 1 epoch -- learning rate of .001
    - 88.31% accuracy after 2 epochs -- learning rate of .0001
VERSION 2:
    - Added one more layer: 784 -> 256 -> 128 -> 64 -> 10
    - learning rate 0.005, 3 epochs --> 94.25% accuracy!
    - 1 min 38s per epoch (still takes a while, mainly due to matrix operations)