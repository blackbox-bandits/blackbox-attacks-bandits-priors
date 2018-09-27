# Prior Convictions: Black-box Adversarial Attacks with Bandits and Priors
This is the code for reproducing the paper "Prior Convictions: Black-box Adversarial Attacks with Bandits and Priors". The paper can be cited as follows:

# Reproducing the results
The results can be reproduced (with the default hyperparameters) with the following command:
```
python main.py [--nes] [--tiling --tile-size 50] --json-config [configs/l2.json | configs/linf.json | configs/linf-nes.json | configs/l2-nes.json] --log-progress
```

You can run ```python main.py --help``` to see all of the available options/hyperparameters.o

You can modify the code where it says "# The results can be analyzed here!" to add custom logging.
