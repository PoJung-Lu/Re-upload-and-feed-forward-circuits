import numpy as np
import pennylane as qml
import jax.numpy as jnp
import jax
from .q_functions import *
from .util import *

from functools import partial
import optax

def model_noise(backend,fake_backend):
    from qiskit_aer.noise import  NoiseModel
    if fake_backend:   #deprecated
        # model_dict = {}
        # i_bknd = backend.index('_')+1
        # f_bknd = 'Fake'+backend[i_bknd].upper()+backend[i_bknd+1:]+'V2'
        # import_str = "from {0} import {1}".format('qiskit.providers.fake_provider',f_bknd)
        # exec(import_str,globals(),model_dict)
        # exec('noise_model = NoiseModel.from_backend({})'.format(f_bknd+'()'),globals(),model_dict)
        # noise_model = model_dict['noise_model']
        print('Fake backend deprecated, falling back to NoiseModel from physical backend.')
        noise_model = NoiseModel.from_backend(backend) 
    else:

        noise_model = NoiseModel.from_backend(backend) 
    return noise_model

def device(num_qubits,noise=False,real_device=False,backend_name=None):#noise_model=None
    if noise:
        from qiskit_ibm_runtime import QiskitRuntimeService
        # QiskitRuntimeService.save_account(channel='ibm_quantum', instance='ibm-q-hub-ntu/ntu-internal/default', token='', overwrite=True)
        service = QiskitRuntimeService()
        '''If backend_name is None, use least_busy backend instead.'''
        if backend_name:
            backend = service.backend(backend_name)  # ibm_strasbourg # 
            print(f'Now using backend: {backend_name}')
        else:
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
            print(f'Now using least busy backend: {backend.name}')
        
        
        if real_device:
            # dev = qml.device('qiskit.remote', wires=<num_qubits>, backend=backend)
            dev = qml.device('qiskit.remote', wires=num_qubits, backend=backend, 
                             provider=provider,initial_layout= [1,2],start_session=True)#)#
        else:
            noise_model = model_noise(backend,fake_backend=False) 
            dev = qml.device('qiskit.aer', wires=num_qubits, diff_method="adjoint", noise_model=noise_model)#
    else: 
        dev = qml.device("default.qubit", wires=num_qubits)
    return dev

def conditional_qnode(f):
    def wrapper(*args):
        self = args[0]
        # if self.noise:
        #     return qml.QNode(f,self.dev)(*args)
        # else: 
        #     return qml.QNode(f,self.dev,interface='jax')(*args)
        return qml.QNode(f,self.dev,interface='jax')(*args)
    return wrapper

class qcircuit:
    # dev = device(num_qubits,noise,real_device,backend,noise_model)
    def __init__(self, *setting, noise=None, real_device=None, backend_name=None, rot='xzx', **kwargs): #params, x, y,
        enc_dim,n_qubits,n_layers,n_reupload,n_rot = setting
        
        
        self.dev = device(n_qubits,noise,real_device,backend_name)#,noise_model
        # self.x = x
        # self.y = y
        self.enc_dim = enc_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_reupload = n_reupload
        self.n_rot = n_rot
        self.noise = noise #kwargs['noise']
        self.rot = rot #kwargs['rot']
        # self.N = len(x[0])                                 # # of examples, and x = X_train.T
        # self.params = params   

        
    def reshape_input(self,x_in): #'''May cause some prob ex. dim_in=4, enc_dim=3, q=5 e.g. not all enc_dim is filled. In this case, the rest of gates are set to 0 '''
        # x_in has dimension  [[x0 vector],[x1 vector],[x2 vector],...] -> (# of features, N)
        enc_dim , n_qubits = self.enc_dim, self.n_qubits
        self.dim_in = len(x_in)
        self.N = len(x_in[0])                                 # # of examples, and x = X_train.T

        if n_qubits*enc_dim >= self.dim_in*2:       # repeat to fill all qubits
            self.rep = n_qubits*enc_dim//self.dim_in
            x_in = jnp.tile(x_in,(self.rep,1))
        if remain_dim:=enc_dim*n_qubits-len(x_in):           # make x_in be integer multiple of enc_dim
            x_in = jnp.vstack((jnp.zeros((remain_dim,self.N)), x_in)) 
        x_in = x_in.reshape(len(x_in)//enc_dim,enc_dim,self.N) # (least_q,enc_dim,N) ## each qubit have enc_dim input dims
        return x_in
    def encode_op(self,x):
        op = ()
        #print(x[0][0].shape)
        for q in range(self.n_qubits):
            for e_i in range(self.enc_dim):
                if even := (e_i+1)%2:   #Rx(x[0]),Rx(x[2])...
                    op = op+(qml.RX(x[q][e_i], wires=q),)#
                    if e_i>=len(x[0]): op = op+(qml.RX(0, wires=q),)  ;print('Warning: # of encoding features is more than enc_dim.')                   
                else:                   #Ry(x[1]),Ry(x[3])...
                    op = op+(qml.RY(x[q][e_i], wires=q),)
                    if e_i>=len(x[0]): op = op+(qml.RY(0, wires=q),)  ;print('Warning: # of encoding features is more than enc_dim.') 
                    # Notice that JAX don't have IndexError: out of bound error.
                    # It simply returns the last results.
        return op

    def reupload_op(self,x):
        op=()
        for q in range(self.n_qubits):
            for e_i in range(self.enc_dim):
                if even := (e_i+1)%2:   #RxRyRxRy...(0,1,2,...)
                    
                    op = op+(qml.RX(x[q][e_i], wires=q),)#x[q][e_i]
                    if e_i>=len(x[0]): op = op+(qml.RX(0, wires=q),)  ;print('Warning: # of encoding features is more than enc_dim.')                   
                else:
                    op = op+(qml.RY(x[q][e_i], wires=q),)
                    if e_i>=len(x[0]): op = op+(qml.RY(0, wires=q),)  ;print('Warning: # of encoding features is more than enc_dim.') 
        return op

    def rot_op(self,p, rot):
        op = ()
        # rot = self.rot
        rots = ["zyz", "xzx", "yzy"]
        if rot not in rots:
            raise ValueError('rotation must be one of {}'.format(rots))
        if rot=='zyz':
            for n_rot in range(self.n_rot):
                for q in range(self.n_qubits):
                    op = op+(qml.Rot(*p[q,n_rot], wires=q),)
                    op = op+(qml.CNOT(wires=[q-1,q]),) if q != 0 else op
        if rot=='xzx':
            for n_rot in range(self.n_rot):
                for q in range(self.n_qubits):
                    op = op+(qml.Rot(*p[q,n_rot], wires=q),)
                    op = op+(qml.CNOT(wires=[q-1,q]),) if q != 0 else op
        if rot=='yzy':
            for n_rot in range(self.n_rot):
                for q in range(self.n_qubits):
                    op = op+(qml.Rot(*p[q,n_rot], wires=q),)
                    op = op+(qml.CNOT(wires=[q-1,q]),) if q != 0 else op
        return op

    # @qml.qnode(dev,interface='jax')   ## To print circuit, use @qml.qnode instead of @conditional_qnode
    @conditional_qnode
    def enc_circuit(self,p, x, rot):
        # p0=p['scaling']
        # p0=jnp.tile(p0,(self.n_qubits,1))
        for r_j in range(self.n_reupload):
            p1 = p['scaling'][r_j][:len(x)]#
            p2 = p['circ'][r_j]
            # p1=jnp.tile(p1.reshape(len(x),self.enc_dim,1),(1,1,self.N))
            x2 = x*p1
            #x2=[x[0],x[1]*p[12],x[2]*p[13],x[3]*p[14],x[4]*p[15],x[5]*p[16]] if problem=='breast_cancer' else [x[:,0],x[:,1]*p[12],x[:,2]*p[13]]
            [qml.apply(op) for op in self.encode_op(x2)]
            [qml.apply(op) for op in self.rot_op(p2, rot)]
            
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
    

    @conditional_qnode
    def circuit(self,p, x, rot):
        #p0 = p['params']
        #x2=[x[0],x[1]*p0[12],x[2]*p0[13]]
        # x1 = jnp.tile(x,(self.enc_dim,1,1)).reshape(self.n_qubits,self.enc_dim,self.N) # (q,N)->(enc,q,N)->(q,enc,N)
        x1 = jnp.tile(x,(self.enc_dim,1)).reshape(self.n_qubits,self.enc_dim) # (q)->(enc,q)->(q,enc)
        
        for r_j in range(self.n_reupload):
            p1 = p['scaling'][r_j]
            p2 = p['circ'][r_j]
            # p1=jnp.tile(p1.reshape(self.n_qubits,self.enc_dim,1),(1,1,self.N))
            x2 = p1*x1
            [qml.apply(op) for op in self.reupload_op(x2)]
            [qml.apply(op) for op in self.rot_op(p2, rot)]
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
    

    # @partial(jax.jit, static_argnums=(0,))
    @conditional_qnode
    def circuit_end(self,p, x, y, n_layers, rot ):
        y=jnp.array(y)

        for r_j in range(self.n_reupload):
            p1 = p['scaling'][r_j]
            p2 = p['circ'][r_j]
            # p1=jnp.tile(p1.reshape(self.n_qubits,self.enc_dim,1),(1,1,self.N))
            # p = p['params']
            # x2=[x[0],x[1]*p[12],x[2]*p[13]]
            if n_layers==1:
                # x2=[x[0],x[1]*p[12],x[2]*p[13],x[3]*p[14],x[4]*p[15],x[5]*p[16]] if problem=='breast_cancer' else [x[0],x[1]*p[12],x[2]*p[13]]
                x2 = x*p1        
                [qml.apply(op) for op in self.encode_op(x2)]
            else:
                # x1 = jnp.tile(x,(self.enc_dim,1,1)).reshape(self.n_qubits,self.enc_dim,self.N) # (q,N)-> enc,q,N)->(q,enc,N)
                x1 = jnp.tile(x,(self.enc_dim,1)).reshape(self.n_qubits,self.enc_dim) # (q)-> enc,q)->(q,enc)
                x2 = x1*p1 
                [qml.apply(op) for op in self.reupload_op(x2)]     
            [qml.apply(op) for op in self.rot_op(p2, rot)]
        return [qml.expval(qml.Hermitian(y, wires=[q])) for q in range(self.n_qubits)] 
        #return qml.expval(qml.Hermitian(y, wires=[0])) ,qml.expval(qml.Hermitian(y, wires=[1]))


    
    @partial(jax.jit, static_argnums=(0,3))
    def j_enc_circuit(self,p, x, rot):
        print('enc_circuit is jitted')
        return jax.vmap(self.enc_circuit,(None,2,None))(p, x, rot)
        
    @partial(jax.jit, static_argnums=(0,3))
    def j_circuit(self,p, x, rot):
        print('circuit is jitted')
        return jax.vmap(self.circuit,(None,1,None))(p, x, rot)
    
    @partial(jax.jit, static_argnums=(0,3,4,5))
    def j_circuit_end(self,p, x, y, n_layers, rot):
        print('circuit_end is jitted')
        if n_layers==1:
            return jax.vmap(self.circuit_end,(None,2,None,None,None))(p, x, y, n_layers, rot)
        else:
            return jax.vmap(self.circuit_end,(None,1,None,None,None))(p, x, y, n_layers, rot)
    
    def qc_nq(self,params,x,y):
        import math
        self.dim_x = (len(x))                              # dimension of x
        self.least_q = math.ceil(self.dim_x/self.enc_dim)       # least qubit # for encoding
        if self.least_q>self.n_qubits: raise ValueError("Number of qubits not enough for encoding features.")
        
        # x = self.reshape_input(self.x)
        # params, y = self.params, self.y
        x = self.reshape_input(x)
        n_layers = self.n_layers
        rot = self.rot
        
        v_enc_circuit = jax.vmap(self.enc_circuit,(None,2,None))
        v_circuit = jax.vmap(self.circuit,(None,1,None))        
        v_circuit_end = jax.vmap(self.circuit_end,(None,2,None,None,None)) if n_layers==1 else jax.vmap(self.circuit_end,(None,1,None,None,None))
        
        for f_i in range(self.n_layers-1):
            #p = params[f_i]
            #print(x.shape)
            p = {i:j[f_i] for i,j in params.items()}
            ocs = v_enc_circuit(p, x, rot) if f_i==0 else v_circuit(p, x, rot)
            # ocs =  self.enc_circuit(self,p, x) if f_i==0 else self.circuit(p, x)
            # print(qml.draw(self.enc_circuit)(self,p, x)) ## To print circuit, use @qml.qnode instead of @conditional_qnode 
            x = jnp.array(ocs)
            
        p = {i:j[self.n_layers-1] for i,j in  params.items()}
        #p = params[self.n_layers-1]
        
        ocs =  v_circuit_end(p, x, y, n_layers, rot)
        
        return ocs
    
    
    # jqc_nq = qc_nq# jax.jit(qc_nq, static_argnums=(0))# 
    def jqc_nq(self,params,x,y):
        import math
        self.dim_x = (len(x))                              # dimension of x
        self.least_q = math.ceil(self.dim_x/self.enc_dim)       # least qubit # for encoding
        if self.least_q>self.n_qubits: raise ValueError("Number of qubits not enough for encoding features.")
        
        # x = self.reshape_input(self.x)
        # params, y = self.params, self.y
        x = self.reshape_input(x)
        n_layers = self.n_layers
        rot = self.rot
        for f_i in range(self.n_layers-1):
            p = {i:j[f_i] for i,j in params.items()}
            ocs = self.j_enc_circuit(p, x, rot) if f_i==0 else self.j_circuit(p, x, rot)
            x = jnp.array(ocs)
        p = {i:j[self.n_layers-1] for i,j in  params.items()}
        # v_circuit_end = jax.vmap(self.circuit_end,(None,2,None,None,None)) if n_layers==1 else jax.vmap(self.circuit_end,(None,1,None,None,None))
        # ocs =  v_circuit_end(p, x, y, n_layers, rot)
        ocs =  self.j_circuit_end(p, x, y, n_layers, rot)
        
        return ocs
    
def test(params, x, y, *args, **kwargs):

    enc_dim,num_qubits,num_layers,num_reupload,num_rot = args     
    
    # for k, v in kwargs.items():
    #     if 'noise' in k:
    #         noise = v
    
    def fidel_function(params, x_i,dm_label, *setting, **kwargs):  
        qc = kwargs['qc']
        if kwargs['noise']:
            return qc.qc_nq(params, x_i, dm_label,)
        else:
            # print('circuit is jitted')
            return qc.jqc_nq(params, x_i, dm_label,)  # qc.jqc_nq()
        
    def jloss(params,fidelities,y_i,shape,num_class,Yc):
        
        b = 1 if shape=="bitwise" else 0     
        #print('fidelities',fidelities.shape)
        return sum([1/2*(params[num_layers+i,j]*fidelities[i,j]-Yc[y_i,2*i*b+j])**2 for  i in range(num_qubits) for j in range(num_class)])    
        #return sum([-(Yc[y_i,2*i*b+j]*jnp.log(fidelities[i,j])) for  i in range(num_qubits) for j in range(num_class)])    
        
    def cost(params, x,y, *setting, **kwargs):
        dm_labels = kwargs['dm_labels']
        num_class = kwargs['num_class_1q']
        shape = kwargs['shape']
        Yc = jnp.array(kwargs['Yc'])
        x_in = x.T   #[[x0 vector],[x1 vector],[x2 vector]]
        loss = 0
        predicted = []
        # qc = qcircuit(*setting, **kwargs)
        # qc = kwargs['qc']
        tot_fidelities = jnp.array([fidel_function(params, x_in, dm, *setting, **kwargs ) for dm in dm_labels]).T #shape:(N,#qubit,#class)
        num_qubits = setting[1]

        
        tot_fidelities = tot_fidelities.reshape(len(y),num_qubits,num_class)
        
        tot_fids = jax.vmap(jnp.kron,(0,0),0)(tot_fidelities[:,0],tot_fidelities[:,1]).reshape(len(y),num_qubits,num_class) if shape=="bitwise" else tot_fidelities

        Loss = jax.vmap(jloss,(None,0,0,None,None,None))(params['loss'],tot_fids,y,shape,num_class,Yc)
        Loss = jnp.sum(Loss)/len(y)/num_qubits
        predicted = jax.vmap(jnp.argmax)(tot_fids) if shape == "bitwise" else jax.vmap(jnp.argmax)(tot_fids)%num_class
        return Loss, predicted

    '''for diff input to same fun., if u take grad after operation it will cause error
       But do operation seperately (for diff input) and add together is ok '''
    '''grad go to 0 if u diff over fun of fun； 
       which is: diff(fun(p)) is ok, but if f=fun(p), diff(f) is 0  
       2022/12/26'''
    (loss,predicted), grad = jax.value_and_grad(cost, has_aux=True)(params,x,y,*args,**kwargs)
    return jnp.array(predicted), loss, grad


def jtest(params, x, y, *args, **kwargs):
    if kwargs['noise']:
        return test(params, x, y, *args, **kwargs)
    else:
        # print(kwargs.keys())
        return jax.jit(test, static_argnums = (4), static_argnames = tuple(kwargs.keys()))(params, x, y, *args, **kwargs)
        
def scores(params, x_tr, y_tr, *args, x_te=None, y_te=None, report=False,  **kwargs):

    predicted_train, loss , grads = jtest(params, x_tr, y_tr, *args, **kwargs)
    accuracy_train = accuracy_score(y_tr, predicted_train)
    
    if x_te is None and y_te is None:
        return accuracy_train, loss  
    else:
        predicted_test, loss_test , grads_test = jtest(params, x_te, y_te, *args, **kwargs)
        accuracy_test = accuracy_score(y_te, predicted_test)
        if report:
            from sklearn.metrics import classification_report
            return classification_report(y_te, predicted_test)
        else:
            return accuracy_train, loss, accuracy_test, loss_test 
        
'''
def fit(params: optax.Params, optimizer: optax.GradientTransformation,
        opt_state,x,y,*args,x_valid=None,y_valid=None,**kwargs) -> optax.Params:
  #opt_state = optimizer.init(params)
  # for k, v in kwargs.items():
  #   if 'noise' in k:
  #       noise = v
  def step(params, opt_state,x,y):
    loss_batches = jnp.array([])
    predicted_train = jnp.array([])
    iter_batch = iterate_minibatches(x, y, batch_size=batch_size)
    for x_batch, y_batch in iter_batch:
        predicted_batch, loss_batch , grads = jtest(params, x_batch, y_batch,*args,**kwargs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_batches = jax.numpy.append(loss_batches,loss_batch)
        predicted_train = jax.numpy.append(predicted_train, predicted_batch)
    loss_value = jnp.average(loss_batches)
    return params, opt_state, loss_value, predicted_train
  n_converge = 0
  
  learning_rate,max_epoch,batch_size,dynamic_size,threshold = kwargs['_h_pm']
  iter_lr = kwargs['iter_lr']
  iter_thres = kwargs['iter_thres']
  lr = kwargs['lr']
  thres_n = kwargs['thres_n']
  

  #loss_history = []
  params_history = []
  state_history = []
  valid_accuracy_history=[]
  ave_loss = 1
  l_r = lr
  thres = thres_n
  for i  in range(max_epoch):
    i=i+1
    state_history.append(opt_state) #save value advance otherwise it saves the updated one
    params_history.append(params)
    
    params, opt_state, loss_value , predicted_train= step(params, opt_state, x, y)
    accuracy_train = accuracy_score(y, predicted_train)
    
    print(f'step {i}, accuracy_train:{accuracy_train}, loss: {loss_value}')
    if x_valid is not None and y_valid is not None:
        accuracy_valid, loss_valid = scores(params, x_valid, y_valid, *args, **kwargs)
        valid_loss_history.append(float(loss_valid))
        valid_accuracy_history.append(float(accuracy_valid))
    loss_history.append(float(loss_value))    
    accuracy_history.append(float(accuracy_train))
    
    if i%dynamic_size==0:
        c_ave = sum(loss_history[-dynamic_size:])/dynamic_size #current loss average
        now_thres = abs(c_ave-ave_loss)/ave_loss
        print(now_thres)
        if  now_thres<=thres:
            try:
                l_r=next(iter_lr)#learning_rate
                #loc_best_accuracy = np.argmax(accuracy_history[-dynamic_size:])-dynamic_size
                loc_best_loss = np.argmin(loss_history[-dynamic_size:])-dynamic_size
                params = params_history[loc_best_loss]
                #print('best known accuracy:', accuracy_history[loc_best_accuracy]) #update lr based on accuracy
                print('accuracy of best loss :', accuracy_history[loc_best_loss]) #update lr based on loss
                
                opt_state.hyperparams['learning_rate'] = l_r
                #print('lr:', l_r, opt_state.hyperparams['learning_rate'])
                try:
                    thres = next(iter_thres)
                except StopIteration:
                    print(thres)
            except StopIteration:
                pass
        print('lr:', l_r, opt_state.hyperparams['learning_rate'])
        print(i)
        n_converge  = n_converge+1 if now_thres<thres_converge else 0
        ave_loss = c_ave
        
    if i==max_epoch or n_converge==max_n_converge:
        n_converge = 1 if n_converge==0 else n_converge 
        #loc_best_accuracy = np.argmax(accuracy_history[-dynamic_size*10:])-dynamic_size*10
        candidate_history = loss_history if x_valid is None else valid_loss_history
        loc_best_loss = np.argmin(candidate_history[-dynamic_size*n_converge:])-dynamic_size*n_converge
        params = params_history[loc_best_loss]
        opt_state = state_history[loc_best_loss]
        num_epoch = i
        print('lr:', l_r, opt_state.hyperparams['learning_rate'])
        print('accuracy of best loss:',accuracy_history[loc_best_loss],loc_best_loss)
        #print('best known accuracy:',accuracy_history[loc_best_loss],loc_best_loss)
        break
  return params, l_r, opt_state, num_epoch
'''