import numpy as np

TOY_STDLIB = {
    "PRM_ROPE_SUB": {
        "subspace_indices": [0, 1, 2, 3],
    },
    "PRM_LOW_RANK_FFN": {
        "W_down": np.array([
            [ 0.5, -0.2,  0.1,  0.0,  0.9, -0.4,  0.3, -0.1],
            [-0.1,  0.6, -0.3,  0.8,  0.2,  0.1, -0.5,  0.2]
        ], dtype=np.float32),
        "W_up": np.array([
            [ 0.1, -0.9],
            [ 0.4,  0.2],
            [-0.3,  0.5],
            [ 0.8, -0.1],
            [ 0.2,  0.6],
            [-0.1, -0.4],
            [ 0.7,  0.3],
            [-0.5,  0.1]
        ], dtype=np.float32)
    }
}
