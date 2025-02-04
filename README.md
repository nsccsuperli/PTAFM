# Polygon Training Architecture for Foundation Model on Dual-Heterogeneous with Network and Device

Large language models have experienced rapid growth, constrained by the computational limits of training foundation models. With the continuous release of new GPU products, high-end devices are increasingly accessible, eventually transitioning into the mid-range and low-end segments. A pivotal focus in current research is the facilitation of joint training across diverse regions and devices. However, this research encounters dual-heterogeneous challenges in both network and device capabilities. 

## Overview 

- We introduce a novel polygonal training architecture for foundation model, designed to support large-scale training paradigms. Our approach incorporates critical factors such as model size, network conditions, and device performance from both global and local perspectives.

- We develop the lightweight polygon initialization algorithm, which considers data centers as the fundamental units from a global perspective. This algorithm assesses computing power, latency, and bandwidth between units to establish an initial training strategy that incorporates both pipeline and data parallelism. 

- We address the complexities introduced by varying combinations of heterogeneous devices and network conditions, which lead to intricate communication scenarios. We design a polygonal local optimization algorithm, which is a precise search strategy. By accurately evaluating communication costs during model training across diverse heterogeneous configurations, we identify an efficient parallel architecture, enabling enhanced collaborative training across devices with fine granularity.






## Environments

- Use TC scripts to control network delay and bandwidth.

You need to prepare different types of devices in advance. In this paper, we used NVIDIA A100 and T4 devices. Meanwhile, you need to set the latency and bandwidth in advance according to different regions and inject them into different nodes.


## Foundation Model Training:

### Download our code:

'''sh
git https://github.com/nsccsuperli/PTAFM.git
'''

### Use the provided Docker environment (Optional, coming soon) 



### Group all nodes based on multiple factors using our lightweight initialization algorithm


'''sh
python ACO_CVRP.py
Required parameters: NUM_STAGE; POWER_SECTION; LATENCY_SECTION; BANDWIDTH_SECTION
'''
- Generate multiple groups

### Use search algorithms to find the optimal strategy 

'''sh
 python GA
'''

- Generate the final pipeline parallel and data parallel groups

### More importantly, Communication optimization mechanism
'''sh
[comm](./comm) directory
'''

### Run script

  
- From each terminal, run cmd:
      
      python dist_runner.py --dist-url tcp://XXX.XXX.XXX.XXX:9000 --world-size N --rank i (i=0,...,N-1)

### Run with Advanced Scripts (recommended)

- Go to the [scripts](./scripts) directory


- bash aws_run_batch_gpt3_optimal.sh 1

## 5 Acknowledgements

We sincerely appreciate the contributions of the following methods
>[DTFM](https://github.com/DS3Lab/DT-FM)
