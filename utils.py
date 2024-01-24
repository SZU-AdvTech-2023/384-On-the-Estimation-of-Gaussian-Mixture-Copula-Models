import numpy as np
import scipy as sc
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
from sklearn import mixture
import math as m
from scipy import interpolate
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# 将数据集划分为训练集，测试集和验证集
def splitData(data, split_ratio=[0.7,0.2,0.1]):
        
    N = data.shape[0]
    
    # 只划分训练集和测试集
    if len(split_ratio) == 2:
        n_training = round(N*split_ratio[0])
        n_valid = round(N*split_ratio[1])
        n_test = 0
    
    elif len(split_ratio) == 3:
        n_training = round(N*split_ratio[0])
        n_valid = round(N*split_ratio[1])
        n_test = round(N*split_ratio[2])
    
    np.random.shuffle(data)
    data_training, data_valid, data_test,_ = np.split(data, np.cumsum([n_training,n_valid,n_test]))
    
    return data_training, data_valid, data_test

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# 通过Chandrupatla算法求根，来计算逆CDF值
def icdf_numerical(u,cdf_funct,lb,ub):
    # 计算逆CDF值可以转换为求u-ψ(x)=0的根
    obj_func = lambda x: cdf_funct(x) - u
    
    # Chandrupatla算法
    x = tfp.math.find_root_chandrupatla(obj_func,low=lb,high=ub)[0]
    return x

# 通过BIC方法来确认最佳GMM及其构件数
# BIC值越小，模型的性能通常更好
def GMM_best_fit(samples,min_ncomp=1,max_ncomp=10, max_iter=200, print_info=False):
    lowest_bic = np.infty
    bic = []
    for n_components in range(min_ncomp, max_ncomp+1):
        # 通过EM算法来求解GMM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full',
                                      reg_covar=1E-4,
                                      max_iter=max_iter,
                                      n_init=5)
        gmm.fit(samples)
        if print_info:
            print('Fittng a GMM on samples with %s components: BIC=%f'%(n_components,gmm.bic(samples)))
        bic.append(gmm.bic(samples))
        # 找到BIC最小的模型
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm    
    return best_gmm

# 标准化高斯混合模型参数
def standardize_gmm_params(alphas,mus,covs,chols=[]):
    weighted_mus = tf.linalg.matvec(tf.transpose(mus),alphas)
    new_mus = mus - weighted_mus
    variances = tf.linalg.diag_part(covs)
    scaling_vec = tf.linalg.matvec(tf.transpose(new_mus**2+variances),alphas)
    scaling_matrix = tf.linalg.diag(1/(scaling_vec**0.5))
    new_mus = tf.linalg.matmul(new_mus,scaling_matrix)
    new_covs=tf.linalg.matmul(tf.linalg.matmul(scaling_matrix,covs),scaling_matrix)
    # 计算新的 Cholesky 分解
    new_chols = tf.linalg.matmul(scaling_matrix,chols) if len(chols) else []
    return alphas,new_mus,new_covs,new_chols

# 参数向量转换为高斯混合模型参数
def vec2gmm_params(n_dims,n_comps,param_vec):
    num_alpha_params = n_comps
    num_mu_params = n_comps*n_dims
    num_sigma_params = int(n_comps*n_dims*(n_dims+1)*0.5)
    
    logit_param, mu_param, chol_param = tf.split(param_vec,[num_alpha_params,num_mu_params,num_sigma_params])
    mu_vectors = tf.reshape(mu_param, shape=(n_comps,n_dims))
    chol_mat_array=tf.TensorArray(tf.float32,size=n_comps)
    cov_mat_array=tf.TensorArray(tf.float32,size=n_comps)
    
    for k in range(n_comps):
        start_idx = tf.cast(k*(num_sigma_params/n_comps),tf.int32)
        end_idx = tf.cast((k+1)*(num_sigma_params/n_comps),tf.int32)
        
        # 从一维的 chol_param 参数中恢复出 Cholesky 矩阵
        chol_mat = tfb.FillScaleTriL(diag_bijector=tfb.Exp()).forward(chol_param[start_idx:end_idx])
        # Cholesky 矩阵与其转置相乘，计算了当前高斯分布的协方差矩阵。
        cov_mat = tf.matmul(chol_mat,tf.transpose(chol_mat))
        chol_mat_array = chol_mat_array.write(k,chol_mat) 
        cov_mat_array =  cov_mat_array.write(k,cov_mat) 
        
    chol_matrices = chol_mat_array.stack()
    cov_matrices = cov_mat_array.stack()     
    return [logit_param,mu_vectors,cov_matrices,chol_matrices]

# 将高斯混合模型参数集合到一个参数向量中
def gmm_params2vec(n_dims,n_comps,alphas,mu_vectors,cov_matrices, chol_matrices=[]):
    param_list = []
    param_list.append(np.log(alphas))
    param_list.append(tf.reshape(mu_vectors,-1))
    """
    对于每个组件,如果提供了chol_matrices,则使用对应的Cholesky矩阵.
    否则,计算cov_matrices的Cholesky分解。
    然后将Cholesky矩阵转换为一维的参数向量,并添加到param_list
    """
    for k in range(n_comps):
        chol_mat=chol_matrices[k] if len(chol_matrices) else tf.linalg.cholesky(cov_matrices[k])
        param_list.append(tfb.FillScaleTriL(diag_bijector=tfb.Exp()).inverse(chol_mat))
    param_vec = tf.concat(param_list,axis=0)
    return param_vec

# 绘制GMCM的密度等高线图
def plotDensityContours(data,log_prob,dim1,dim2):
    mins=np.min(data,axis=0)
    maxs=np.max(data,axis=0)
    ngrid=100
    X,Y=np.meshgrid(np.linspace(mins[dim1],maxs[dim1],ngrid),(np.linspace(mins[dim2],maxs[dim2],ngrid)))
    X=X.astype('float32')
    Y=Y.astype('float32')
    z=np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],axis=1)
    
    # 计算GMCM密度值
    prob_z=np.exp(log_prob(z).numpy())
    Z=prob_z.reshape(ngrid,ngrid)

    plt.contour(X,Y,Z,20)
    plt.plot(data[:,dim1],data[:,dim2],'ko')
    plt.xlabel(f'dim_{dim1}',fontsize=14)
    plt.ylabel(f'dim_{dim2}',fontsize=14)

# 通过kmeans++来初始化GMCM参数
def kmeans_init(data, k):
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(data)
    
    # 均值
    means = kmeans.cluster_centers_
    
    # 计算权重
    labels = kmeans.labels_
    weights = np.zeros(k)
    for i in range(k):
        weights[i] = np.sum(labels == i) / data.shape[0]

    # 计算协方差
    covariances = np.zeros((k, data.shape[1], data.shape[1]))
    for i in range(k):
        diff = (data[labels == i] - kmeans.cluster_centers_[i]).T
        covariances[i] = np.dot(weights[i] * diff, diff.T) / np.sum(labels == i)
    
    return weights.astype('float32'), means.astype('float32'), covariances.astype('float32')