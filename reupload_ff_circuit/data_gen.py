##########################################################################
#Quantum classifier
#Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil, J. Ignacio Latorre
#Code by APS
#Code-checks by ACL
#June 3rd 2019


#Universitat de Barcelona / Barcelona Supercomputing Center/Institut de Ciències del Cosmos

###########################################################################

## This file creates the data points for the different problems to be tackled by the quantum classifier 



import numpy as np

problems = ['breast_cancer','moon','circle', '3 circles', 'wavy circle', 'hypersphere', 'tricrown', 'non convex', 'crown', 'sphere', 'squares', 'wavy lines']

# Default sample counts for different problems
DEFAULT_SAMPLES = {
    'sphere': 4500,
    'hypersphere': 5000,
    'breast_cancer': 569
}

def data_generator(problem, samples=None, noise=0.1):
    """
    This function generates the data for a problem using optimized dictionary dispatch.

    INPUT:
        -problem: Name of the problem, one of: 'circle', '3 circles', 'hypersphere', 'tricrown',
                  'non convex', 'crown', 'sphere', 'squares', 'wavy lines', 'moon', 'breast_cancer'
        -samples: Number of samples for the data (None for defaults)
        -noise: Noise level for 'moon' and 'breast_cancer' problems
    OUTPUT:
        -data: set of training and test data
        -settings: things needed for drawing
    """
    problem = problem.lower()
    if problem not in problems:
        raise ValueError('problem must be one of {}'.format(problems))

    # Set default samples using dictionary lookup
    if samples is None:
        samples = DEFAULT_SAMPLES.get(problem, 4200)

    # Validation for breast_cancer
    if problem == 'breast_cancer' and samples > 569:
        raise ValueError('number of samples must be less or equal to 569')

    # Dictionary-based dispatch for O(1) lookup
    PROBLEM_GENERATORS = {
        'circle': _circle,
        '3 circles': _3_circles,
        'wavy lines': _wavy_lines,
        'squares': _squares,
        'sphere': _sphere,
        'non convex': _non_convex,
        'crown': _crown,
        'tricrown': _tricrown,
        'hypersphere': _hypersphere,
        'moon': _moon,
        'breast_cancer': _breast_cancer,
        'wavy circle': _circle,  # Assuming 'wavy circle' maps to _circle
    }

    generator = PROBLEM_GENERATORS[problem]

    # Handle generators with different signatures
    if problem in ['moon', 'breast_cancer']:
        data, settings = generator(samples, noise)
    else:
        data, settings = generator(samples)

    return data, settings 

def _breast_cancer(samples, noise):
    from sklearn.datasets import load_breast_cancer
    from sklearn.utils import shuffle
    bc_data = load_breast_cancer()
    X, y = bc_data.data, bc_data.target
    X, y = shuffle(X, y, random_state=np.random.randint(samples))
    selected_features =  [0, 1, 4, 5, 11]
    data = []
    for i,j in zip(X[:samples,selected_features], y[:samples]):
        data.append([i,j])
    return data, None

def _moon(samples, noise):
    from sklearn.datasets import make_moons
    x, y = make_moons(samples, noise=noise)
    data=[]
    for i,j in zip(x,y):
        data.append([i,j])
    return data, None

def _circle(samples):
    centers = np.array([[0, 0]])
    radii = np.array([np.sqrt(2/np.pi)])
    data=[]
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y = 0
        for c, r in zip(centers, radii):  
            if np.linalg.norm(x - c) < r:
                y = 1 

        data.append([x, y])
            
    return data, (centers, radii)

def _3_circles(samples):
    centers = np.array([[-1, 1], [1, 0], [-.5, -.5]])
    radii = np.array([1, np.sqrt(6/np.pi - 1), 1/2]) 
    data=[]
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y = 0
        for j, (c, r) in enumerate(zip(centers, radii)): 
            if np.linalg.norm(x - c) < r:
                y = j + 1 
                
        data.append([x, y])
    return data, (centers, radii)
    

def _wavy_lines(samples, freq = 1):
    def fun1(s):
        return s + np.sin(freq * np.pi * s)
    
    def fun2(s):
        return -s + np.sin(freq * np.pi * s)
    data=[]
    dim=2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[1] < fun1(x[0]) and x[1] < fun2(x[0]): y = 0
        if x[1] < fun1(x[0]) and x[1] > fun2(x[0]): y = 1
        if x[1] > fun1(x[0]) and x[1] < fun2(x[0]): y = 2
        if x[1] > fun1(x[0]) and x[1] > fun2(x[0]): y = 3        
        data.append([x, y])

    return data, freq

def _squares(samples):
    data=[]
    dim=2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[0] < 0 and x[1] < 0: y = 0
        if x[0] < 0 and x[1] > 0: y = 1
        if x[0] > 0 and x[1] < 0: y = 2
        if x[0] > 0 and x[1] > 0: y = 3        
        data.append([x, y])
    
    return data, None


def _non_convex(samples, freq = 1, x_val = 2, sin_val = 1.5):
    def fun(s):
        return -x_val * s + sin_val * np.sin(freq * np.pi * s)
    
    data = []
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[1] < fun(x[0]): y = 0
        if x[1] > fun(x[0]): y = 1
        data.append([x, y])

    return data, (freq, x_val, sin_val)
            
def _crown(samples):
    c = [[0,0],[0,0]]
    r = [np.sqrt(.8), np.sqrt(.8 - 2/np.pi)]
    data = []
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if np.linalg.norm(x - c[0]) < r[0] and np.linalg.norm(x - c[1]) > r[1]:
            y = 1
        else: 
            y=0
        data.append([x, y])

    return data, (c, r)


def _tricrown(samples):
    centers = [[0,0],[0,0]]
    radii = [np.sqrt(.8 - 2/np.pi), np.sqrt(.8)]
    data = []
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y=0
        for j,(r,c) in enumerate(zip(radii, centers)):
            if np.linalg.norm(x - c) > r:
                y = j + 1
        data.append([x, y])

    return data, (centers, radii)

def _sphere(samples):
    centers = np.array([[0, 0, 0]]) 
    radii = np.array([(3/np.pi)**(1/3)]) 
    data=[]
    dim = 3
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        y = 0
        for c, r in zip(centers, radii): 
            if np.linalg.norm(x - c) < r:
                y = 1 

        data.append([x, y])
    
    return data, (centers, radii)

def _hypersphere(samples):
    centers = np.array([[0, 0, 0, 0]]) 
    radii = np.array([(2/np.pi)**(1/2)]) 
    data=[]
    dim = 4
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1 
        y = 0
        for c, r in zip(centers, radii): 
            if np.linalg.norm(x - c) < r:
                y = 1 

        data.append([x, y])
    
    return data, (centers, radii)


