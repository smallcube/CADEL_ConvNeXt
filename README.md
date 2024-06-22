# CADEL_ConvNeXt
This is the combination of our CADEL and ConvNeXt

You can run this code with "torchrun --nproc_per_node n main_convnext_ensemble_pc.py", where n is the number of GPUs in your server.


If you find our idea or code inspiring, please cite our paper:

```bibtex
@article{CADEL,
  title={Long-tailed Classification via CAscaded Deep Ensemble Learning},
  author={Zhi Chen, Jiang Duan, Yu Xiong, Cheng Yang and Guoping Qiu},
  year={2023},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

This code also combines [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), if you use this code, please also cite：

```bibtex
@inproceedings{liu2022convnet,
  title={A convnet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11976--11986},
  year={2022}
}
```
