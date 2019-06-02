#  Flappybird-in-RL
<img src="https://github.com/Dingrong123/Flappybird-in-RL/blob/master/assets/sprites/FBgif1.gif" width="250" height="550" alt="图片加载失败时，显示这段字"/>
##  Model
It is a classical DQN with off-policy strategy. There are 2 networks. One is exploring network and another is target network.

## Training tricks
*  The model will converge faster if the background of the game is replaced by a totally black one. 
*  Similarly, I remove the picture of scores in the game window to make it trains faster.
* If the probability of random choice is set too high at beginning, the bird will flappy too much that stay at the top. So I initialize the probability of random choice to 0.1 , which is quite small compared to training other games.
* Besides,I give high penalty if the bird flies high and knocks on the pipe to prevent the above condition.
* The reward of going through pipes is increased to  encourage the bird to act properly.

## Training process
It takes me 3 whole days to train the model on the desktop

## Disclaimer
This work is highly based on the following repos:<br>
1. [sourabhv/FlapPyBird](https://github.com/sourabhv/FlapPyBird)
2.  https://github.com/yenchenlin/DeepLearningFlappyBird
