The code of paper [Dynamic Graph Convolutional Networks with Spatiotemporal Missing Pattern Awareness](https://ieeexplore.ieee.org/abstract/document/10887757)



### Datasets

- PEMSBAY
- Weather
- BeijingAir

### How to run

- Train

```bash
python main.py --dataset PEMS --mask_ratio 0.2 --epochs 200
```

- Test

```bash
python test.py --dataset PEMS --mask_ratio 0.2 --exp_name xxx # xxx is the timestamp when training starts
```



### Citation

```python
@INPROCEEDINGS{10887757,
  author={Pang, Bingheng and Liang, Zhuoxuan and Li, Wei and Zheng, Xiangping and Abdein, Rokia},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Dynamic Graph Convolutional Networks with Spatiotemporal Missing Pattern Awareness}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Adaptation models;Correlation;Graph convolutional networks;Time series analysis;Signal processing;Market research;Imputation;Spatiotemporal phenomena;Forecasting;Speech processing;Multivariate time series forecasting;Missing patterns;Dynamic graphs;Time series},
  doi={10.1109/ICASSP49660.2025.10887757}}
```



