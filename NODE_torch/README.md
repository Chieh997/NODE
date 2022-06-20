# Neral Differential Equation

Differential equations and neural networks have been two dominant modelling paradigms in recent years. Neural differential equations(NDEs) combined these two approach and build a continuous-depth model. NDEs consider a input network as a vector field of a differential equation and embed it into a larger network. Parameterized that differential equations and solve it with data-driven way like other neural network models.

This project contains some research on NDEs and reproduce one of the experiment result.

## Paper Summery
These notes were posted on HackMD in the beginning. It's more recommended to view them on HackMD for better experiences.
+ *Neural Ordinary Differential Equations* [1]\
   The vanilla mothod for NODEs. Consider the parameters in the differential equations as constant.
   + [[Summery]](https://github.com/Chieh997/Demo/blob/main/NODE_torch/Paper_summary/2018.NODEs.md)[[HackMD]](https://hackmd.io/@Chieh997/BygEi-iaY) [[Presentation]](https://hackmd.io/@Chieh997/2017NODEs)
+ *Dissecting Neural ODEs* [2]\
   Establish a gereral framework for NODEs. Parameters in the differential equations are regared as functions of time. Also consider different augmentated strategy and  how controlled over input networks. Develop several variant models based on these concepts.
   + [[Summery]](https://github.com/Chieh997/Demo/blob/main/NODE_torch/Paper_summary/2020.DissectingNODEs.md)[[HackMD]](https://hackmd.io/@Chieh997/rJqo7jKaK) [[Presentation]](https://hackmd.io/@Chieh997/2020DissNODEs)
+ *Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise* [3]\
   Apply some of the most usable regularization mechanisms in discrete networks and develop a new framwork called NSDEs.
   + [[Summery]](https://github.com/Chieh997/Demo/blob/main/NODE_torch/Paper_summary/2019.NSDEs.md)[[HackMD]](https://hackmd.io/@Chieh997/SJrX-pcTF)

## Experiment: Image Classification
This experiment (Stefano, 2020) is to analysis different augmentation strategies for NODEs.
+ Setting difference:\
 Due to equipment limitations, the training epoch is set to be smaller (20 to 12).

|           | NODE | ANODE | IL-NODE |
| ----      | ---  | ---   | ----    |
| Test. Acc.| 96.88| 98.94 | 99.25|
| Avg. NFE  | 153  | 162   | 86.5 |
| Param.[K] | 16.9 | 20.8  | 21.1 |

 + Comparations and Probelems\
 The testing accuracies are similar to the original results. We can still see the performance improvements for augmented stratergies. \
 But the number of parameters are diffenent, especially for NODE who is significantly smaller than other methods. Also, the NFEs, interpreted as an implicit number of layers, are higher than the desired ones.
 
---
## References
[[1]](https://arxiv.org/abs/1806.07366) R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural Ordinary Differential Equations,” *Neural Information Processing Systems*, 2018. 

[[2]](https://arxiv.org/abs/2002.08071) M. Stefano , P. Michael , P. Jinkyoo , Y. Atsushi , and A. Hajime , “Dissecting Neural ODEs,” *Advances in Neural Information Processing Systems*, vol. 33, 2020.‌

[[3]](https://arxiv.org/abs/1906.02355) X. Liu, T. Xiao, S. Si, Q. Cao, S. Kumar, and C.-J. Hsieh, “Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise,” *arXiv:1906.02355 [cs, stat]*, Jun. 2019.
