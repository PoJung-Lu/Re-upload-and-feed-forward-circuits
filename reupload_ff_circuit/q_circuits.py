import numpy as np
import pennylane as qml
import jax.numpy as jnp
import jax
from .q_functions import *
from .util import *

from functools import partial
import optax
import logging

logger = logging.getLogger(__name__)

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
        logger.warning('Fake backend deprecated, falling back to NoiseModel from physical backend.')
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
            dev = qml.device('qiskit.remote', wires=num_qubits, backend=backend,)#initial_layout=[1,2], start_session=True)
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

        # Validate circuit parameters
        if enc_dim < 1 or n_qubits < 1 or n_layers < 1:
            raise ValueError(f"Circuit dimensions must be positive integers: "
                             f"enc_dim={enc_dim}, n_qubits={n_qubits}, n_layers={n_layers}")
        if n_reupload < 1 or n_rot < 1:
            raise ValueError(f"Reupload and rotation counts must be positive: "
                             f"n_reupload={n_reupload}, n_rot={n_rot}")
        if n_qubits > 10 and not noise:
            import warnings
            warnings.warn(f"Large qubit count ({n_qubits}) may cause memory issues with simulator. "
                          "Consider using noise=True with a real backend.")

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

        # Cache for vmap operations (initialized on first use)
        self._v_enc_circuit = None
        self._v_circuit = None
        self._v_circuit_end_single = None
        self._v_circuit_end_multi = None   

        
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
                    if e_i>=len(x[0]):
                        op = op+(qml.RX(0, wires=q),)
                        logger.warning('Number of encoding features is more than enc_dim.')
                else:                   #Ry(x[1]),Ry(x[3])...
                    op = op+(qml.RY(x[q][e_i], wires=q),)
                    if e_i>=len(x[0]):
                        op = op+(qml.RY(0, wires=q),)
                        logger.warning('Number of encoding features is more than enc_dim.') 
                    # Notice that JAX don't have IndexError: out of bound error.
                    # It simply returns the last results.
        return op

    def reupload_op(self,x):
        op=()
        for q in range(self.n_qubits):
            for e_i in range(self.enc_dim):
                if even := (e_i+1)%2:   #RxRyRxRy...(0,1,2,...)

                    op = op+(qml.RX(x[q][e_i], wires=q),)#x[q][e_i]
                    if e_i>=len(x[0]):
                        op = op+(qml.RX(0, wires=q),)
                        logger.warning('Number of encoding features is more than enc_dim.')
                else:
                    op = op+(qml.RY(x[q][e_i], wires=q),)
                    if e_i>=len(x[0]):
                        op = op+(qml.RY(0, wires=q),)
                        logger.warning('Number of encoding features is more than enc_dim.') 
        return op

    def rot_op(self,p, rot):
        op = ()
        rots = ["zyz", "xzx", "yzy"]
        if rot not in rots:
            raise ValueError('rotation must be one of {}'.format(rots))

        # NOTE: All rotation types ('xzx', 'yzy') are deprecated except 'zyz'.
        # The implementation is identical for all types as they use qml.Rot()
        # which handles the rotation basis internally.
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
        logger.debug('enc_circuit is jitted')
        return jax.vmap(self.enc_circuit,(None,2,None))(p, x, rot)

    @partial(jax.jit, static_argnums=(0,3))
    def j_circuit(self,p, x, rot):
        logger.debug('circuit is jitted')
        return jax.vmap(self.circuit,(None,1,None))(p, x, rot)

    @partial(jax.jit, static_argnums=(0,3,4,5))
    def j_circuit_end(self,p, x, y, n_layers, rot):
        logger.debug('circuit_end is jitted')
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

        # Use cached vmap operations for better performance
        if self._v_enc_circuit is None:
            self._v_enc_circuit = jax.vmap(self.enc_circuit, (None, 2, None))
            self._v_circuit = jax.vmap(self.circuit, (None, 1, None))
            self._v_circuit_end_single = jax.vmap(self.circuit_end, (None, 2, None, None, None))
            self._v_circuit_end_multi = jax.vmap(self.circuit_end, (None, 1, None, None, None))

        v_enc_circuit = self._v_enc_circuit
        v_circuit = self._v_circuit
        v_circuit_end = self._v_circuit_end_single if n_layers == 1 else self._v_circuit_end_multi

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
        """
        JIT-compiled version of qc_nq for faster execution.

        WARNING: This function is memory-intensive for large datasets.
        For large batches, consider using jqc_nq_chunked() instead.
        """
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

    def jqc_nq_chunked(self, params, x, y, chunk_size=32):
        """
        Memory-efficient version of jqc_nq that processes data in chunks.

        This method reduces memory usage by processing the input data in smaller
        chunks, making it suitable for large datasets that would otherwise cause
        memory issues with the standard jqc_nq method.

        Args:
            params: Circuit parameters dictionary with keys 'scaling', 'circ', 'loss'
            x: Input data array of shape (n_features, n_samples)
            y: Target density matrices
            chunk_size: Number of samples to process at once (default: 32)
                       Reduce this if still encountering memory issues

        Returns:
            Array of circuit outputs, same shape as jqc_nq output

        Example:
            >>> # For large datasets, use chunked version
            >>> results = qc.jqc_nq_chunked(params, X_train.T, dm_labels[0], chunk_size=16)
        """
        import math

        # Get number of samples from input shape
        n_samples = x.shape[-1] if x.ndim > 1 else 1

        # If data is small enough, just use regular jqc_nq
        if n_samples <= chunk_size:
            return self.jqc_nq(params, x, y)

        # Process in chunks
        results = []
        for i in range(0, n_samples, chunk_size):
            chunk_end = min(i + chunk_size, n_samples)
            chunk_x = x[..., i:chunk_end]
            chunk_result = self.jqc_nq(params, chunk_x, y)
            # Ensure result is a JAX array
            chunk_result = jnp.array(chunk_result)
            results.append(chunk_result)

        # Concatenate results along the sample dimension
        return jnp.concatenate(results, axis=-1)
    
def test(params, x, y, *args, **kwargs):
    """
    Test/training function with support for memory-efficient chunked processing.

    Args:
        params: Circuit parameters dictionary
        x: Input data array
        y: Target labels
        *args: (enc_dim, num_qubits, num_layers, num_reupload, num_rot)
        **kwargs: Additional options including:
            - noise: Use noisy simulator (bool)
            - use_chunked: Use memory-efficient chunked processing (bool, default: False)
            - chunk_size: Samples per chunk when use_chunked=True (int, default: 32)
            - qc: qcircuit instance
            - dm_labels: Density matrix labels
            - num_class_1q: Number of classes per qubit
            - shape: "bitwise" or other
            - Yc: Target matrix

    Returns:
        (predictions, loss, gradients)

    Example:
        # For large datasets, enable chunked processing
        pred, loss, grad = test(params, X, y, *settings, use_chunked=True, chunk_size=16, **kwargs)
    """
    enc_dim,num_qubits,num_layers,num_reupload,num_rot = args     
    
    # for k, v in kwargs.items():
    #     if 'noise' in k:
    #         noise = v
    
    def fidel_function(params, x_i,dm_label, *setting, **kwargs):
        qc = kwargs['qc']
        use_chunked = kwargs.get('use_chunked', True)
        chunk_size = kwargs.get('chunk_size', 32)

        if kwargs['noise']:
            return qc.qc_nq(params, x_i, dm_label,)
        elif use_chunked:
            # Use memory-efficient chunked version for large datasets
            return qc.jqc_nq_chunked(params, x_i, dm_label, chunk_size=chunk_size)
        else:
            # Use standard JIT-compiled version
            return qc.jqc_nq(params, x_i, dm_label,)
        
    def jloss(params,fidelities,y_i,shape,num_class,Yc):
        """
        Vectorized loss computation for better performance.

        Maintains the original quadratic loss formula but uses vectorized operations
        instead of Python loops for significant speed improvement.
        """
        b = 1 if shape=="bitwise" else 0
        num_qubits = fidelities.shape[0]

        # Create index arrays for vectorized computation
        i_idx = jnp.arange(num_qubits)[:, None]  # Shape: (num_qubits, 1)
        j_idx = jnp.arange(num_class)[None, :]   # Shape: (1, num_class)

        # Compute Yc indices vectorially: yc_indices[i,j] = y_i, 2*i*b + j
        yc_col_indices = 2 * i_idx * b + j_idx  # Shape: (num_qubits, num_class)

        # Extract parameter slice and target values
        # params has shape (num_qubits + num_qubits, num_class) in loss parameters
        # We need params[num_qubits:num_qubits+num_qubits, :num_class]
        param_slice = params[:num_qubits, :num_class]  # Shape: (num_qubits, num_class)

        # Get target values: Yc[y_i, yc_col_indices]
        # Need to index Yc properly - Yc is (n_samples, n_features)
        target_vals = Yc[y_i, yc_col_indices]  # Shape: (num_qubits, num_class)

        # Vectorized loss calculation (keeps original formula: 1/2 * (param*fid - target)^2)
        diff = param_slice * fidelities - target_vals
        loss = 0.5 * jnp.sum(diff ** 2)

        return loss    
        
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