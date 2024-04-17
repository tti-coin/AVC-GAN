# iTransGAN

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. Train and evaluate the AutoEncoder. 

```
python run.py --is_ae_training 1 --is_gan_training 0 --is_syntheting 0
```

2. Train and ebaluate WGAN-GP with trained AE.

3. Generate synthetized data.
