# Dataset
the bootstrapping dataset can be downloaded from the yan et al.(2020)[End-to-end bootstrapping neural network for entity set expansion].

the pre-training dataset can be find [here](https://drive.google.com/file/d/1Ow6Rf_LIilKvm0dVuJSF5dGigTMosOQq/view?usp=sharing)

# Pre-training and Fine-tuning
## pre-training
### self-supervised
```bash
python pretrain_self.py --out_model_file models/xxx1
```
### supervised
```bash
python pretrain_sup.py --input_model_file models/xxx1 --out_model_file models/xxx2
```
## fine-tuing
to run the model please input like
```bash
python fine_tune.py --input_model_file models/xxx2
```
