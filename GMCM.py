import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
import time
import utils as utl
from bijectors import Marginal_transform, GMM_bijector
from sklearn import mixture
import matplotlib as plt

# 定义GMC类
class GMC:
    def __init__(self, n_dims, n_comps, param_vec=None):
        self.ndims = n_dims
        self.ncomps = n_comps   
        # 参数包括权重alpha, 均值mu, 方差cov（方差为对称矩阵，只保存上三角或者下三角）
        self.total_trainable_params = int(n_comps*(1+n_dims+0.5*n_dims*(n_dims+1))) # alpha, mu, covs
        self.params = param_vec
        if self.params is not None:
            assert tf.size(self.params) == self.total_trainable_params, 'the supplied parameter vector is not commensurate with the n_dims, and n_comps'
            self.params = param_vec
    
    # 定义GMC变换分布
    @property
    def distribution(self):
        # 将参数向量转为GMM参数，包括权重，均值，方差（以及Cholesky矩阵）
        logits,mus,covs,chols = utl.vec2gmm_params(self.ndims,self.ncomps,self.params)
        # 初始化双射映射器
        gmc_bijector = GMM_bijector(self.ndims, self.ncomps, [logits, mus, tf.linalg.diag_part(covs)])
        # 基分布：GMM
        base_dist = tfd.MixtureSameFamily(tfd.Categorical(logits=logits),
                                          tfd.MultivariateNormalTriL(loc=mus,scale_tril=chols))
        # 将GMC分布指定为变换分布
        gmc_dist = tfd.TransformedDistribution(distribution=base_dist,bijector=gmc_bijector)    
        return gmc_dist   
    
    # 计算两个先验高斯项（论文中公式有错误，报告中已经修改过来）
    @property
    def identifiability_prior(self):
        # 将GMM参数转为参数向量
        logits,mus,covs,_ = utl.vec2gmm_params(self.ndims,self.ncomps,self.params)        
        alphas = tf.math.softmax(logits)
        variances = tf.linalg.diag_part(covs)        
        vec1 = tf.linalg.matvec(tf.transpose(mus),alphas)
        vec2 = tf.linalg.matvec(tf.transpose(variances + mus**2),alphas)
        log_prior_1 = tfd.MultivariateNormalDiag(loc=tf.zeros(self.ndims),scale_diag=1E-1*tf.ones(self.ndims)).log_prob(vec1)
        log_prior_2 = tfd.MultivariateNormalDiag(loc=tf.ones(self.ndims) ,scale_diag=1E-1*tf.ones(self.ndims)).log_prob(vec2)
        return log_prior_1,log_prior_2
    
    # 初始化GMC参数
    # 可以通过'random','gmm','params','kmeans++'这四种方法进行初始化
    def init_params(self,initialization=['random',None]):
        init_method=initialization[0]
        # 随机初始化
        if init_method == 'random':
            seed_val=initialization[1]
            if seed_val is not None:
                np.random.seed(seed_val)
            alphas = tf.ones(self.ncomps)/self.ncomps
            mus = tf.constant(np.random.randn(self.ncomps,self.ndims).astype('float32'))
            covs = tf.repeat(tf.expand_dims(tf.eye(self.ndims),0),self.ncomps,axis=0)
            chols = tf.repeat(tf.expand_dims(tf.eye(self.ndims),0),self.ncomps,axis=0)
        # 通过GMM来进行初始化
        elif init_method == 'gmm':            
            data=initialization[1]
            gmm = mixture.GaussianMixture(n_components=self.ncomps,
                                          covariance_type='full',
                                          max_iter=1000,
                                          n_init=5)
            gmm.fit(data)
            alphas = gmm.weights_.astype('float32')
            mus = gmm.means_.astype('float32')
            covs = gmm.covariances_.astype('float32')
        elif init_method == 'kmeans':
            data=initialization[1]
            alphas, mus, covs = utl.kmeans_init(data, self.ncomps)
            
        # 通过输入参数进行初始化                  
        elif init_method == 'params':
            alphas,mus,covs=initialization[1]
        
        # 标准化参数
        alphas,mus,covs,_ = utl.standardize_gmm_params(alphas,mus,covs)
        
        # 将参数转为参数向量进行初始化
        param_vec = tf.Variable(utl.gmm_params2vec(self.ndims,self.ncomps,alphas,mus,covs))
        self.params=param_vec
    
    def fit_dist(self,
                 u_mat_train,   # 训练数据
                 n_comps,       # 组件个数     
                 u_mat_valid=None,  # 验证数据
                 optimizer = tf.optimizers.Adam(learning_rate=1E-3),    # 优化器
                 max_iters = 1000,      # 最大迭代次数
                 batch_size = 10,       # 批量大小
                 print_interval=100,    # 打印间隔
                 regularize=True,       # 是否加入正则项
                 plot_results = False,  # 是否绘制结果
                 return_param_updates=False):   # 是否返回参数更新
        
        self.ncomps = n_comps
        
        # 初始化GMC参数
        if self.params is None:
            self.init_params()
        
        # 训练步骤
        @tf.function
        def train_step(u_selected):
            with tf.GradientTape() as tape:
                neg_gmc_ll = -tf.reduce_mean(self.distribution.log_prob(u_selected))
                ident_prior = self.identifiability_prior
                if regularize:  # 带有正则项
                    total_cost = neg_gmc_ll - tf.reduce_sum(ident_prior)
                else:
                    total_cost = neg_gmc_ll
            # 计算目标函数关于模型的梯度，并使用优化器更新模型参数
            grads = tape.gradient(total_cost, self.params)
            if not (tf.reduce_any(tf.math.is_nan(grads)) or tf.reduce_any(tf.math.is_inf(grads))):
                optimizer.apply_gradients(zip([grads], [self.params])) # 更新GMC参数
            return total_cost
        
        # 验证步骤，计算验证数据的目标函数值
        @tf.function
        def valid_step(u_valid):
            neg_gmc_ll = -tf.reduce_mean(self.distribution.log_prob(u_valid))
            return neg_gmc_ll - tf.reduce_sum(self.identifiability_prior) if regularize else neg_gmc_ll

        # 初始化记录训练的过程的变量，包括训练和验证的目标函数值，参数历史记录以及早停参数
        neg_ll_trn = np.empty(max_iters)  
        neg_ll_trn[:] = np.NaN
        neg_ll_vld = np.empty(max_iters)  
        neg_ll_vld[:] = np.NaN
        params_history=np.zeros((max_iters,self.params.shape[0]))
        patience,last_vld_err=0,float('inf')
        
        ts = time.time() # 开始时间
        # 优化迭代过程
        for itr in np.arange(max_iters):
            if patience>5: break # 提前停止迭代
            # 从训练数据中随机选择一批数据
            np.random.seed(itr)
            samps_idx = np.random.choice(u_mat_train.shape[0],batch_size,replace=False)
            # 执行训练步骤并记录结果
            u_selected_trn = tf.gather(u_mat_train,samps_idx)
            neg_ll_trn[itr] = train_step(u_selected_trn).numpy()
            # 如果启用了参数更新返回，记录当前的参数值
            if return_param_updates: params_history[itr,:]=self.params.numpy()
            # 每隔一定的迭代次数，打印当前的训练信息
            if tf.equal(itr%print_interval,0) or tf.equal(itr,0):
                if u_mat_valid is not None: 
                    # 如果连续多次迭代验证误差都增加，那么提前停止迭代
                    neg_ll_vld[itr] = valid_step(u_mat_valid).numpy()
                    if neg_ll_vld[itr]>last_vld_err: 
                        patience+=1
                    else:
                        patience=0
                    last_vld_err=neg_ll_vld[itr]
                       
                time_elapsed = np.round(time.time()-ts,1)
                print(f'@ Iter:{itr}, \
                        Training error: {np.round(neg_ll_trn[itr],1)}, \
                        Validation error: {np.round(neg_ll_vld[itr],1)}, \
                        Time Elapsed: {time_elapsed} s')    
        # 绘制结果
        if plot_results:
            plt.plot(neg_ll_trn)
            plt.plot(neg_ll_vld)
            plt.xlabel('Iteration',fontsize=12)
            plt.ylabel('Neg_logLike',fontsize=12)
            plt.legend(['train'],fontsize=12)
         
        return neg_ll_trn, neg_ll_vld, params_history

# 定义GMCM类
class GMCM:
    def __init__(self, 
                 n_dims, 
                 data_transform=None, 
                 marginals_list=None, 
                 gmc=None):
        
        self.ndims = n_dims
        self.preproc_transform=data_transform   
        self.marg_dists=marginals_list      # 各个维度的边缘分布
        self.gmc = gmc
        
        # 如果已经指定了边缘分布，则定义边缘双射器
        if self.marg_dists is not None:
            self.marg_bijector = Marginal_transform(self.ndims,self.marg_dists)
            
        # 验证gmc参数是否与指定的gmcm一致 
        if self.gmc:
            assert n_dims==self.gmc.ndims, 'GMC object dimensions should match the specified GMCM dimension'
            self.ncomps = self.gmc.ncomps
    
    # 返回GMCM的变换分布，包括预先学习的边缘分布以及基GMM边缘分布。
    # 如果指定了预处理变换，那么就会在变换链中包含该变换。
    @property
    def distribution(self):
        if self.preproc_transform:
            gmcm_dist = tfd.TransformedDistribution(distribution=self.gmc.distribution.distribution,
                                                    bijector=tfb.Chain([self.preproc_transform,
                                                                        self.marg_bijector,
                                                                        self.gmc.distribution.bijector]))
        else:
            gmcm_dist = tfd.TransformedDistribution(distribution=self.gmc.distribution.distribution,
                                                    bijector=tfb.Chain([self.marg_bijector,
                                                                        self.gmc.distribution.bijector]))
        return gmcm_dist
    
    # 学习边缘分布
    def learn_marginals(self,max_allowable_comps=10):
        marg_dist_list=[]
        # 对每个维度，从训练数据中选择对应数据，通过GMM来拟合这些数据（用BIC准则来确定最优构件）
        for j in range(self.ndims):
            input_vector = self.data_trn[:,j].reshape(-1,1)
            marg_gmm_obj = utl.GMM_best_fit(input_vector,
                                            min_ncomp=1,
                                            max_ncomp=max_allowable_comps)
            marg_gmm_tfp = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=marg_gmm_obj.weights_.flatten().astype('float32')),
                components_distribution=tfd.Normal(loc=marg_gmm_obj.means_.flatten().astype('float32'),
                scale = marg_gmm_obj.covariances_.flatten().astype('float32')**0.5),)
            # 创建一个包含各种信息的字典
            info_dict={'cdf':marg_gmm_tfp.cdf,
                       'log_pdf':marg_gmm_tfp.log_prob,
                       'lb':tf.reduce_min(input_vector)-3*tfp.stats.stddev(input_vector),
                       'ub':tf.reduce_max(input_vector)+3*tfp.stats.stddev(input_vector)                         
                      }
            
            marg_dist_list.append(info_dict)     
        return marg_dist_list
        
    
    def fit_dist_likelihood(self,                        
                     data_trn,      # 训练数据
                     n_comps,       # 组件个数
                     data_vld=None, # 验证数据
                     optimizer = tf.optimizers.Adam(learning_rate=1E-3),    # 优化器
                     init = 'random',   # 初始化方法
                     max_iters = 1000,  # 最大迭代次数
                     batch_size = 10,   # 批量大小
                     print_interval=100,    # 打印间隔
                     regularize=True,       # 是否加入正则项
                     plot_results = False): # 是否绘制结果
        
        # 如果指定了预处理变换，则对训练数据和验证数据进行变换
        if self.preproc_transform: 
            data_trn = self.preproc_transform.inverse(data_trn).numpy() 
            if data_vld is not None: 
                data_vld = self.preproc_transform.inverse(data_vld).numpy()
        
        self.data_trn = data_trn
        self.data_vld = data_vld
        
        # 如果没有学习了边缘分布，则调用learn_marginals方法来学习边缘分布
        if self.marg_dists is None:
            print('Learning Marginals')
            ts = time.time()
            self.marg_dists = self.learn_marginals()
            print(f'Marginals learnt in {np.round(time.time()-ts,2)} s.') 
        
        # 根据学到的边缘分布来初始化边缘变换
        self.marg_bijector = Marginal_transform(self.ndims,self.marg_dists) 
        
        self.ncomps = n_comps
        # 进行x到u的变换
        u_mat_trn = self.marg_bijector.inverse(self.data_trn)

        # 初始化GMC对象
        gmc_obj=GMC(self.ndims,self.ncomps)
        
        # 初始化GMC参数
        if init=='random':
            initialization=['random',None]
        elif init=='gmm':
            initialization=['gmm',data_trn]
        elif init=='kmeans':
            initialization=['kmeans',data_trn]
        elif isinstance(init, list):
            initialization=['params',init]
        gmc_obj.init_params(initialization)
        
        # 估计GMC参数
        neg_ll_trn,neg_ll_vld,param_history=gmc_obj.fit_dist(u_mat_trn,
                                                             n_comps, 
                                                             u_mat_valid=data_vld,
                                                             optimizer = optimizer, 
                                                             max_iters = max_iters, 
                                                             batch_size = batch_size, 
                                                             print_interval = print_interval, 
                                                             regularize = regularize, 
                                                             plot_results = plot_results)
        self.gmc=gmc_obj
        
        return neg_ll_trn, neg_ll_vld, param_history
    
    # 返回沿指定维度的边缘GMCM模型
    def get_marginal(self,dim_list):
        logits,mus,covs,_ = utl.vec2gmm_params(self.ndims,self.ncomps,self.gmc.params)
        # 根据维度索引选择参数
        alphas = tf.math.softmax(logits)
        mus_new = tf.gather(mus, dim_list, axis=1)
        covs_new = tf.TensorArray(tf.float32,self.ncomps)
        for k in range(self.ncomps):
            temp_mat = covs[k].numpy()
            covs_new = covs_new.write(k,temp_mat[np.ix_(dim_list,dim_list)])
        covs_new = covs_new.stack()
        
        # 将新的参数转为参数向量，并用这个参数向量来初始化一个新的GMC对象
        marginal_gmc_params = utl.gmm_params2vec(len(dim_list),self.ncomps,alphas,mus_new,covs_new)
        marg_gmc = GMC(len(dim_list),self.ncomps,marginal_gmc_params)
        
        # 从原始的边缘分布列表中选择指定的维度得到新的边缘分布列表
        marg_list_new = []
        for j in range(self.ndims):
            if j in dim_list:
                marg_list_new.append(self.marg_dists[j])
                
        # 创建新的GMCM对象
        marg_gmcm_dist = GMCM(len(dim_list), 
                              data_transform=self.preproc_transform, 
                              marginals_list=marg_list_new, 
                              gmc=marg_gmc)
        return marg_gmcm_dist   
    
    # 返回给定数据（X）的分量（C）的后验概率，即P（C|X）
    def predict_prob(self,X):
        assert X.shape[1] == self.ndims, 'Dimension mismatch. The input data should have the shape n_samples x n_dims'
        # 提取基分布（即GMM）
        base_gmm = self.distribution.distribution
        # 将输入数据映射到基空间
        Z = self.distribution.bijector.inverse(X)
        # 计算给定数据组件后验分布
        posterior = base_gmm.posterior_marginal(Z)
        posterior_prob = tf.math.softmax(posterior.logits_parameter(),axis=1).numpy()
        return posterior_prob

    # 返回给定数据（X）的预测分量
    def predict(self,X):
        # 得到给定数据X的分量概率
        posterior_probs = self.predict_prob(X)
        # 概率最大的分量即为所属分量
        comp_label = np.argmax(posterior_probs,axis=1).astype('int32')
        return comp_label