## GMCM

使用 TensorFlow-Probability (TFP) 构造实现高斯混合 Copula 模型 (GMCM)。 GMCM 是高斯混合模型 (GMM) 的更具表现力的替代方案，同时具有相同的参数化来编码多模态依赖结构。

- 参考论文：Tewari, Ashutosh. "On the estimation of Gaussian mixture copula models." *International Conference on Machine Learning*. PMLR, 2023.

- 论文网址：https://proceedings.mlr.press/v202/tewari23a.html

> @inproceedings{tewari2023estimation,
>   title={On the estimation of Gaussian mixture copula models},
>   author={Tewari, Ashutosh},
>   booktitle={International Conference on Machine Learning},
>   pages={34090--34104},
>   year={2023},
>   organization={PMLR}
> }

### Requirements

- python>=3.7
- numpy>=1.19.5
- TensorFlow>=2.5.0
- TensorFlow-Probability>=0.13.0
- scikit-learn>=0.23.2
- pandas>=1.1.3

### 代码结构

- bijectors.py：实现了两个双射器，包括预先学习的边缘bijector以及GMM bijector.
- GMCM.py: 实现了GMCM模型。包括GMCM类和GMC类。
- utils.py：实现了多个可复用的功能函数。
- main.py：主函数。实现了GMCM参数估计以及可视化结果。

### 如何运行

1. 安装 Python

   我们的代码需要 Python 3.7 或更高版本，可以从 Python 官方网站下载并安装。

2. 创建虚拟环境
   虚拟环境可以帮助你管理项目的依赖项，避免不同项目之间的依赖冲突。

3.  安装依赖性
   需要安装以下依赖项，可以通过 pip 来安装。
   • python>=3.7
   • numpy>=1.19.5
   • TensorFlow>=2.5.0
   • TensorFlow-Probability>=0.13.0
   • scikit-learn>=0.23.2
   • pandas>=1.1.3

4.  运行代码
   最后，可以运行文件夹中的代码。确保所有代码文件（如 main.py）在当前目录下。然
   后，可以在命令行中使用 Python 来运行你的代码：

   ```
   python main.py
   ```

5.  修改数据集（可选）
   在 main.py 中可以修改需要的数据集，要注意的是，如果不需要对数据进行对数变换的
   预处理，请注释 main.py 的第 42 行，取消第 43 行的注释。

   