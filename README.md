#  Flappybird-in-RL

##  Model
It is a classical DQN with off-policy strategy. There are 2 networks. One is exploring network and another is target network.

## Training tricks
* In github I notice someone says that the model will converge faster if the background of the game is replaced by a totally black one. I tried it and it seems true.
*  Similarly, I remove the picture of scores in the game window to make it trains faster.
* If the probability of random choice is set too high at beginning, the bird will flappy too much that stay at the top. So I initialize the probability of random choice to 0.1 , which is quite small compared to training other games.
* Besides,I give high penalty if the bird flies high and knocks on the pipe to prevent the above condition.
* The reward of going through pipes is increased to  encourage the bird to act properly.

## Training process
It takes me 3 whole days to train the model on the desktop
