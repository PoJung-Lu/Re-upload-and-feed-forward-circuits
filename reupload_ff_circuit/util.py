from .data_gen import data_generator
from .q_functions import *
# from .q_circuits import *
import jax
import jax.numpy as jnp
import numpy as np
def setting_generator(num_settings,start_values):
    settings, results= [], []
    r_enc, r_q ,r_f, r_r, r_rot = num_settings 
    e0, q0, f0, r0, rot0 = start_values 
    for e in range(r_enc):
        e = e0+e
        for q in range(r_q):
            q = q0+q 
            for f in range(r_f):
                f = f0+f 
                for r in range(r_r):
                    r = r0+r
                    for n_rot in range(r_rot):
                        n_rot = rot0+n_rot 
                        settings.append([e,q,f,r,n_rot] )
    return settings


def data_gen(problem,num_training=300, seed_num = 42):
    np.random.seed(seed_num)
    data=data_generator(problem,num_training)[0]
    x=np.array([i[0] for i in data])#.astype(float)
    y=np.array([i[1] for i in data])#.astype(float)
    return x,y #torch.tensor(x),torch.tensor(y)
    
def plot_data(x, y, fig=None, ax=None):
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    brown = y == 0
    reds = y == 1
    green = y == 2
    blues = y == 3
    ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
    ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
    ax.scatter(x[green, 0], x[green, 1], c="green", s=20, edgecolor="k")
    ax.scatter(x[brown, 0], x[brown, 1], c="brown", s=20, edgecolor="k")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a.tolist()
    
#@jax.jit
def accuracy_score(y_true, y_pred):
    score = y_true == y_pred
    return score.sum().astype(int) / len(y_true)

def iterate_minibatches(inputs, targets, batch_size):
    for start_idx in range(0, inputs.shape[0], batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]
        

        
def initialize_data(problem,n_training=None,n_test=None,seed_num=0,enc_dim=3,**kwargs):
    from sklearn.model_selection import train_test_split
    for k,v in kwargs.items():
        if 'preprocess' in k:
            preprocess = v
        
    def normalize(x,preprocess='scaling'):
        from sklearn import preprocessing  
        if preprocess:
            preprocess = preprocess.lower()
            preprocesses = ["standardization", "scaling", "normalization"]
            if preprocess not in preprocesses:
                raise ValueError(f'preprocess must be one of {preprocesses}')
            if preprocess == 'standardization':
                x = preprocessing.StandardScaler().fit(x).transform(x)*2*np.pi
            if preprocess == 'scaling':
                x = preprocessing.MinMaxScaler().fit_transform(x)*2*np.pi
            if preprocess == 'normalization':
                x = preprocessing.normalize(x, norm='l2')*2*np.pi
        return x

    if problem=='breast_cancer':
        if n_training>569:
            raise ValueError('num_training must < 569')
        # if n_test: global num_test # in order to change the global value of num_test
        if n_test > 569-n_training:
            n_test = 569-n_training
            print('Warning: num_test is set to', n_test)
        x,y = data_gen(problem,569,seed_num=seed_num)
        Xtrain, Xtest, y_train, y_test = train_test_split(
            x,y,test_size=n_test,train_size=n_training,
            random_state=seed_num,stratify=y)
    else:
        Xtrain, y_train = data_gen(problem,n_training,seed_num=seed_num)
        Xtest, y_test = data_gen(problem,n_test,seed_num=np.random.randint(1e6))
    
    X_train = Xtrain
    X_test = Xtest
        
    X_train = normalize(X_train,preprocess)
    X_test = normalize(X_test,preprocess)
    

    if n_training and n_test:
        return X_train, y_train, X_test, y_test
    else:
        if n_training:
            return X_train, y_train
        else:
            return X_test, y_test
# enc_dim,num_qubits,num_layers,num_reupload,num_rot
def initialize_params(enc_dim, n_qubits, n_feedforward, n_reupload, n_rot, num_class, seed_num=40): #num_class= number of class to be classified for one qibit
    num_layers = n_feedforward
    key = jax.random.PRNGKey(seed_num)
    key_s, key_c, key_l = jax.random.split(key,3)
    p_s = jax.random.uniform(key_s,(n_feedforward, n_reupload, n_qubits, enc_dim),dtype=jnp.float64) #12*params,2*scaling factor  
    p_c = jax.random.uniform(key_c,(n_feedforward, n_reupload, n_qubits, n_rot, 3),dtype=jnp.float64)
    p_l = jax.random.uniform(key_l,(n_qubits, num_class),dtype=jnp.float64)
    p = {'scaling':p_s, 'circ':p_c,'loss':p_l}
    return p

def plot_loss_history(l_h, val_l_h=[], setting=None, fig_name=''):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    plt.plot(range(len(l_h)),l_h,color=(255/255,100/255,100/255),label='Loss')
    if val_l_h:
        plt.plot(range(len(val_l_h)),val_l_h,color='g',label='Validation loss')
    plt.title("Loss") # title
    plt.ylabel("loss") # y label
    plt.xlabel("epoch") # x label
    plt.legend()
    plt.show()
    if fig_name:
        fig.savefig('{}.png'.format(fig_name), transparent=True)
    elif fig_name is None:
        pass  #don't save, just print
    else:    
        fig.savefig('{}_loss.png'.format(setting), transparent=True)        