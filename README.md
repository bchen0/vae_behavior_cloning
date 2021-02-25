# VAE Behavior Cloning

This repository contains code used for the experiments in the paper 
"Behavior Cloning in Atari Games Using a Combined Variational Autoencoder and
Predictor Model". There are three 
components to this:

1)  Generating trajectories.
2)  Training the Combined Variational Encoder (VAE) / Predictor, as well as the 
alternative behavior cloning models, on the trajectory data.
    - The three behavior cloning models are:
        - The Combined VAE / Predictor.
        - A discriminative model (CNN) that is based off the player.
        - A separate VAE that first extracts a latent space representation and 
        a separate predictor that takes the latent space as input.
3)  Testing the trained imitator models on gameplay.

Due to different package requirements, 1 and 3 are done in the "gameplay" 
directory. 2) is performed in 
this main directory. The requirements for the main directory and in gameplay
are listed in the requirements.txt file in each directory.

__Instructions for Generating Trajectory Data__: 

Trajectories can be generated
for the stored DQN Models in gameplay/dqn_models. These models are trained using
stable baselines. While the code is written in particular for DQN models, other
models from stable baselines should be usable as well, with a few edits to 
gameplay/trajectories.py.

python trajectories.py ENV_NAME will generate trajectories for the 
specified Atari game in the default location, using the DQN model for that game
from gameplay/dqn_models. Possible options for ENV_NAME are: 
- BeamRiderNoFrameskip-v4
- BreakoutNoFrameskip-v4
- EnduroNoFrameskip-v4
- MsPacmanNoFrameskip-v4
- PongNoFrameskip-v4
- QbertNoFrameskip-v4
- SeaquestNoFrameskip-v4
- SpaceInvadersNoFrameskip-v4


__Instructions for Training Imitators__:

Running python run_experiments.py SAVE_PATH ENV_NAME will train the three 
behavior cloning models. The code assumes that trajectory data has been 
generated already and is located in gameplay/trajectories. Already-trained 
imitators can be found in gameplay/trained_imitator_models.

__Instructions for Testing Imitators on Gameplay__:

The testing procedure described in Algorithm 1 is performed in
gameplay/test_on_gameplay. The code uses the already-trained imitator models 
in gameplay/trained_imitator_models, although they can be replaced by models
trained in the previous section.
