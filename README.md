# MEIDTM
Title : Instance-Dependent Label-Noise Learning with Manifold-Regularized Transition Matrix Estimation.  

## Abstract

​	In label-noise learning, estimating the transition matrix has attracted more and more attention as the matrix plays an important role in building statistically consistent classifiers. However, it is very challenging to estimate the transition matrix *T*(**x**), where **x** *denotes the instance, because it is unidentifiable under the instance-dependent noise (IDN). To address this problem, we have noticed that, there are psychological and physiological evidences showing that we humans are more likely to annotate instances of similar appearances to the same classes, and thus poor-quality or ambiguous instances of similar appearances are easier to be mislabeled to the nearby or same noisy classes. Therefore, we propose assumption on the* geometry *of* *T*(**x**) that “the closer two instances are, the more similar their corresponding transition matrices will be”. More specifically, we formulate above assumption into the manifold embedding, to effectively reduce the degree of freedom of  *T*(**x**) and make it stably estimable. This proposed manifold-regularized technique works by directly reducing the estimation error without hurting the approximation error about the estimation problem of *T*(**x**). Experimental evaluations on four synthetic and two real-world datasets demonstrate our method is superior to state-of-the-art approaches for label-noise learning on IDN.

## Dependencies
We implement our methods by PyTorch on NVIDIA RTX 3090 Ti. The environment is as bellow:
- [Ubuntu 20.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version >= 1.9.0
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 11.1
- [Anaconda3](https://www.anaconda.com/)

## Experiments
We verify the effectiveness of MEIDTM on four  instance-dependent label noisy datasets (F-MNIST, CIFAR-10, CIFAR-100 and SVHN), and two real-world noisy dataset (clothing1M and Food101N).       Here is an example: 

```bash
python run_ours.py --dataset cifar10 --noise_rate 0.3 --lam 0.3
```

If you find this code useful in your research, please cite :

```bash
@inproceedings{ ,
  title={Instance-Dependent Label-Noise Learning with Manifold-Regularized Transition Matrix Estimation.},
  author={ },
  booktitle={ },
  year={ }
}
```
