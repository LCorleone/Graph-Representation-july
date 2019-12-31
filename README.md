CogDL-july
======
The Tensorflow Implementation based on [CogDL](https://github.com/THUDM/cogdl)  
This is just a simple reimplementation for practice!
Note that this task is still actively under development, so feedback and contributions are welcome. The results seem to differ from the CogDL, hope for any ideas!
****
	
|Author|LCorleone|
|---|---
|E-mail|lcorleone@foxmail.com


****
## Requirements
* tensorflow 2
* keras 2.3.1
* networkx


## Usage
* All parameters can be set in para_config.py.
* In main.py, set model, dataset and task and run main.py.

## Support
* Unsupervised node classification for dataset: blogcatalog, PPI and Wikipedia.
* node classification for dataset: cora.

## Results
* Unsupervised node classification (Micro-F1 0.9)
****
|Algorithm|wikipedia|blogcatalog
|---|---|---
|line|0.464|Todo
|netmf|0.418|0.352
|grarep|Todo|0.386
|hope|Todo|0.376
|deepwalk|Todo|0.388
|node2vec|Todo|0.387
|prone|Todo|0.425
****

* node classification (Acc)
****
|Algorithm|cora
|---|---
|GCN|0.81
****

## Reference
[keras-gcn](https://github.com/tkipf/keras-gcn)

