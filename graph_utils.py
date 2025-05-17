import numpy as np

def generate_sparse_precision(p, graph_type='chain', sparsity=0.1):
    """Generate sparse positive-definite precision matrix based on a graph."""
    precision = np.eye(p)

    if graph_type == 'chain':
        for i in range(p - 1):
            precision[i, i+1] = precision[i+1, i] = -0.3
    elif graph_type == 'random':
        for i in range(p):
            for j in range(i+1, p):
                if np.random.rand() < sparsity:
                    precision[i, j] = precision[j, i] = np.random.uniform(-0.4, -0.1)

    precision += p * np.eye(p)
    return precision
