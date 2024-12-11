import numpy as np
import jax.numpy as jnp
def density_matrix(state):
    return state * np.conj(state).T

def state_labels(theta, phi):
    label_0 = np.array([[1.], [0.]])
    label_1 = np.array([[0.], [1.]])
    return [np.cos(np.array(a)/2)*label_0+np.exp(np.array(b)*1j)*np.sin(np.array(a)/2)*label_1 for a,b in zip(theta, phi)]

        
#Bloch sphere representation: 
#psi = alpha|0> + beta|1> = cos(theta/2)+e^(i*phi)sin(theta/2)

def angles(shape):
    shapes = ["square plane", "tetrahedron", "bitwise","binary"]
    shape = shape.lower()
    if shape not in shapes:
        raise ValueError('problem must be one of {}'.format(shapes))
    if shape == "square plane":
        theta = [np.pi/2]*4
        phi = [np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4]
    if shape == "tetrahedron":
        X = [[0, 0], [0,1], [1,0],[1,1]]
        enc = []
        for x in X:
            MyEnc = [None, None]
            if x[0] or x[1]:
                #MyEnc[0] = 2*np.arccos(1/np.sqrt(3))
                MyEnc[0] = np.arccos(-1/3)
            else:
                MyEnc[0] = 0
            MyEnc[1] = (x[0]*2 + x[1])*2*np.pi/3
            enc.append(MyEnc)
            Enc = np.array(enc).T
        theta, phi = Enc[0], Enc[1]
    if shape in ["bitwise","binary"]:
        theta = [0,np.pi]
        phi = [0,0]
    return theta, phi
def plot_class_states(dm_labels,theta, phi, bloch3d=False):
    import qutip
    if bloch3d:
        b = qutip.Bloch() 
        b3d = qutip.Bloch3d()  #need Mayavi,PyQt
        for i in state_labels(theta, phi):
            s=(i[0,0]*qutip.basis(2, 0) + i[1,0]*qutip.basis(2, 1)).unit()
            b.add_states(s)
            b3d.add_states(s)
        b.show()
        b3d.show()
    else:
        b = qutip.Bloch() 
        for i in state_labels(theta, phi):
            s=(i[0,0]*qutip.basis(2, 0) + i[1,0]*qutip.basis(2, 1)).unit()
            b.add_states(s)
        b.show()
        # if 3d has err: pip install mayavi PyQt5 
        # and see https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris
        # or https://askubuntu.com/questions/1418016/glibcxx-3-4-30-not-found-in-conda-environment
        
def yc(c_states,num_class,shape,num_qubits=2): #mutual fidelities of c_states
    
    Yc = jnp.array([(np.abs(i.T.conj().dot(j)))**2 for i in c_states for j in c_states])
    Yc = Yc.reshape([num_class,num_class])
    if shape == "bitwise":
        for i in range(num_qubits-1):
            Yc = jnp.kron(Yc, Yc)
    return Yc

def predefined_states_dm(shape,num_qubits,display=True):
    from .util import totuple
    theta, phi = angles(shape)
    num_class_1q = len(theta)     # of class for one qubit
    print('theta =',theta,'\nphi =',phi)
    c_states = jnp.array(state_labels(theta, phi))
    print('c_states=',c_states,'\nshape:',c_states.shape)
    dm_labels = jnp.array([density_matrix(s) for s in c_states])  
    if display=='3d':
        plot_class_states(dm_labels,theta, phi, bloch3d=True)  
    elif display:
        plot_class_states(dm_labels,theta, phi, bloch3d=False)  
    Yc = yc(c_states,num_class_1q,shape,num_qubits)
    dm_labels = totuple(dm_labels)
    
    return c_states,dm_labels, Yc

