# Global Bootstrapping Neural Network for Entity Set Expansion
The source codes for EMNLP:fingdings(2020) paper:[Global Bootstrapping Neural Network for Entity Set Expansion](https://www.aclweb.org/anthology/2020.findings-emnlp.331.pdf)
## Dataset
the bootstrapping dataset can be downloaded from the yan et al.(2020)[End-to-end bootstrapping neural network for entity set expansion](https://aaai.org/ojs/index.php/AAAI/article/view/6482/6338) or directly from [Google Driver](https://drive.google.com/file/d/1Ow6Rf_LIilKvm0dVuJSF5dGigTMosOQq/view?usp=sharing).

the pre-training dataset can be find [here](https://drive.google.com/file/d/1CulQu5oixrhBev4ECFhTQARiaCghwHtM/view?usp=sharing)

## Pre-training and Fine-tuning
### pre-training
#### self-supervised
```bash
python pretrain_self.py --output_model_file models/xxx1
```
#### supervised
```bash
python pretrain_sup.py --input_model_file models/xxx1 --output_model_file models/xxx2
```
### fine-tuing
to run the model please input like
```bash
python fine_tune.py --input_model_file models/xxx2
```
## Citation
Please cite the following paper if you find our code is helpful, please cite:

```bibtex
@inproceedings{yan-etal-2020-global,
    title = "Global Bootstrapping Neural Network for Entity Set Expansion",
    author = "Yan, Lingyong  and
      Han, Xianpei  and
      He, Ben  and
      Sun, Le",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.331",
    doi = "10.18653/v1/2020.findings-emnlp.331",
    pages = "3705--3714"
}
```
