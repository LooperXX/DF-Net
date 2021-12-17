# Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dynamic-fusion-network-for-multi-domain-end/task-oriented-dialogue-systems-on-kvret)](https://paperswithcode.com/sota/task-oriented-dialogue-systems-on-kvret?p=dynamic-fusion-network-for-multi-domain-end)

This repository contains the PyTorch implementation of the paper: 

**Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog**. [Libo Qin](http://ir.hit.edu.cn/~lbqin/), [Xiao Xu](http://ir.hit.edu.cn/~xxu/), [Wanxiang Che](http://ir.hit.edu.cn/~car/), [Yue Zhang](https://frcchang.github.io/), [Ting Liu](http://ir.hit.edu.cn/~liuting/). ***ACL 2020***. [[PDF]](https://www.aclweb.org/anthology/2020.acl-main.565/)

If you use any source codes or the datasets included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

<pre>
@inproceedings{qin-etal-2020-dynamic,
    title = "Dynamic Fusion Network for Multi-Domain End-to-end Task-Oriented Dialog",
    author = "Qin, Libo  and
      Xu, Xiao  and
      Che, Wanxiang  and
      Zhang, Yue  and
      Liu, Ting",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.565",
    pages = "6344--6354",
    abstract = "Recent studies have shown remarkable success in end-to-end task-oriented dialog system. However, most neural models rely on large training data, which are only available for a certain number of task domains, such as navigation and scheduling. This makes it difficult to scalable for a new domain with limited labeled data. However, there has been relatively little research on how to effectively use data from all domains to improve the performance of each domain and also unseen domains. To this end, we investigate methods that can make explicit use of domain knowledge and introduce a shared-private network to learn shared and specific knowledge. In addition, we propose a novel Dynamic Fusion Network (DF-Net) which automatically exploit the relevance between the target domain and each domain. Results show that our models outperforms existing methods on multi-domain dialogue, giving the state-of-the-art in the literature. Besides, with little training data, we show its transferability by outperforming prior best model by 13.9{\%} on average.",
}
</pre>
![contrast](img/contrast.png)

In the following, we will guide you how to use this repository step by step.

## Architecture

<div align=center><img src="img/framework.png"  alt="framework" width="300" height="500"  /></div>

## Results

![result](img/result.png)

> We clean our code, rerun the experiments based on the following environment and the suggested hyper-parameter settings.

|Datasets|BLEU|F1|Navigate F1|Weather F1|Calendar F1|Datasets|BLEU|F1|Restaurant F1|Attraction F1|Hotel F1|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|SMD|15.2|62.5|55.7|57.3|73.8|MultiWOZ|9.5|34.8|37.5|31.2|32.8|

## Preparation

Our code is based on PyTorch 1.2 Required python packages:

-   numpy==1.14.2
-   tqdm==4.44.1
-   pytorch==1.2.0
-   python==3.6.3
-   cudatoolkit==9.2
-   cudnn==7.6.5

We highly suggest you using [Anaconda](https://www.anaconda.com/) to manage your python environment.

## How to Run it

The script **myTrain.py** acts as a main function to the project, you can run the experiments by the following commands.

```Shell
# SMD dataset
python myTrain.py -gpu=True -ds=kvr -dr=0.2 -bsz=32 -tfr=0.8 -an=SMD -op=SMD.log
# MultiWOZ 2.1 dataset
python myTrain.py -gpu=True -ds=woz -dr=0.2 -bsz=32 -tfr=0.9 -an=WOZ -op=WOZ.log
```

We also provide our reported model parameters in the `save/best` directory, you can run the following command to evaluate them and so on.

```SHELL
python myTrain.py -gpu=True -e=0 -ds=kvr -bsz=32 -path=save/best/SMD -op=SMD.log
python myTrain.py -gpu=True -e=0 -ds=woz -bsz=32 -path=save/best/MultiWOZ -op=WOZ.log
```

Due to some stochastic factors(e.g., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to reproduce the results reported in our paper. 

All the hyper-parameters are in the `utils/config.py` and here are the suggested hyper-parameter settings for grid search:

-   Dropout ratio [0.1, 0.15, 0.2, 0.25, 0.3]
-   Batch size [8, 16, 32]
-   Teacher forcing ratio [0.7, 0.8, 0.9, 1.0]

If you have any question, please issue the project or [email](mailto:xxu@ir.hit.edu.cn) me and we will reply you soon.

## Acknowledgement

**Global-to-local Memory Pointer Networks for Task-Oriented Dialogue**. [Chien-Sheng Wu](https://jasonwu0731.github.io/), [Richard Socher](https://www.socher.org/), [Caiming Xiong](http://www.stat.ucla.edu/~caiming/). ***ICLR 2019***. [[PDF]](https://arxiv.org/abs/1901.04713) [[Open Reivew]](https://openreview.net/forum?id=ryxnHhRqFm) [[Code]](https://github.com/jasonwu0731/GLMP)

>   We are highly grateful for the public code of GLMP!