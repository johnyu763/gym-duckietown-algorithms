Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:1
glxinfo
cd learning
python3 -m reinforcement.pytorch.train_ppo --model-dir=/pvcvolume/
