# data/ 目录

Kaggle ASL Alphabet 数据集解压到这里，预期结构：

```
data/
  asl_alphabet_train/
    A/  B/  C/ ... Z/   (每类一个文件夹)
  asl_alphabet_test/    (Kaggle 自带的，很少，不作为我们的 test)
```

> 我们真正的 test = 自采 never-before-seen 数据，放 data/selfcollected/
> 大文件不要提交到 git（见 .gitignore）。
