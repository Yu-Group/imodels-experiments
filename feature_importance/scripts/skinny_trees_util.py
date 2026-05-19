"""
Small SkinnyTrees helper for the feature-importance simulations.

Adapted from mazumder-lab/SkinnyTrees:
https://github.com/mazumder-lab/SkinnyTrees

Original code is MIT licensed, Copyright (c) 2024 Mazumder Lab.
"""

import numpy as np


def fit_skinny_trees_regressor(
        X,
        y,
        num_trees=50,
        max_depth=3,
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        kernel_l2=1.0,
        kernel_constraint=100.0,
        anneal=True,
        temperature=0.01,
        validation_split=0.0,
        patience=None,
        random_state=27,
        verbose=0):
    """
    Fit a SkinnyTrees-style sparse soft tree ensemble and return feature scores.

    The score is the row norm of the learned split-weight matrix, matching the
    feature-support extraction used in the SkinnyTrees regression scripts.
    """
    try:
        import tensorflow as tf
    except Exception as err:
        raise ImportError(
            "SkinnyTrees requires tensorflow. TensorFlow failed to import in "
            "this environment; check the tensorflow/numpy versions."
        ) from err

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_state)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1 and y.shape[1] != 1:
        raise ValueError("SkinnyTrees helper supports univariate regression only.")
    y = y.reshape(-1, 1)
    n_features = X.shape[1]

    class ConstantLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, learning_rate):
            super().__init__()
            self.learning_rate = learning_rate

        def __call__(self, step):
            return tf.convert_to_tensor(self.learning_rate, dtype=tf.float32)

        def get_config(self):
            return {"learning_rate": self.learning_rate}

    class ProximalGroupL0(tf.keras.constraints.Constraint):
        def __init__(self, lr, lam=0.0, use_annealing=False, temperature=0.1):
            super().__init__()
            self.lr = lr
            self.lam = lam
            self.use_annealing = use_annealing
            self.temperature = temperature
            self.iterations = tf.Variable(0, trainable=False, dtype=tf.int64)

        def __call__(self, w):
            self.iterations.assign_add(1)
            lam_lr = self.lam * self.lr(self.iterations)
            if self.use_annealing:
                scheduler = 1.0 - tf.exp(
                    -tf.cast(self.temperature, w.dtype)
                    * tf.cast(self.iterations, w.dtype)
                )
            else:
                scheduler = 1.0

            w_norm = tf.norm(w, ord="euclidean", axis=1, keepdims=True)
            threshold = tf.sqrt(2.0 * tf.cast(lam_lr * scheduler, w.dtype))
            return tf.where(w_norm > threshold, w, tf.zeros_like(w))

        def get_config(self):
            return {
                "lam": self.lam,
                "use_annealing": self.use_annealing,
                "temperature": self.temperature,
            }

    class TreeEnsembleWithGroupSparsity(tf.keras.layers.Layer):
        def __init__(
                self,
                num_trees,
                max_depth,
                leaf_dims,
                activation="sigmoid",
                node_index=0,
                internal_eps=0.0,
                kernel_regularizer=None,
                kernel_constraint=None):
            super().__init__()
            self.num_trees = num_trees
            self.max_depth = max_depth
            self.leaf_dims = leaf_dims
            self.node_index = node_index
            self.internal_eps = internal_eps
            self.leaf = node_index >= 2 ** max_depth - 1
            self.max_split_nodes = 2 ** max_depth - 1
            self.activation = tf.keras.activations.get(activation)
            self.kernel_regularizer = kernel_regularizer
            self.kernel_constraint = kernel_constraint

            if not self.leaf:
                if self.node_index == 0:
                    self.dense_layer = tf.keras.layers.Dense(
                        self.num_trees * self.max_split_nodes,
                        kernel_regularizer=self.kernel_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        activation=None,
                    )
                masking = np.zeros((1, self.num_trees, self.max_split_nodes))
                masking[:, :, self.node_index] = 1
                self.masking = tf.constant(masking, dtype=self.dtype)
                self.left_child = TreeEnsembleWithGroupSparsity(
                    self.num_trees,
                    self.max_depth,
                    self.leaf_dims,
                    activation=activation,
                    node_index=2 * self.node_index + 1,
                    internal_eps=internal_eps,
                )
                self.right_child = TreeEnsembleWithGroupSparsity(
                    self.num_trees,
                    self.max_depth,
                    self.leaf_dims,
                    activation=activation,
                    node_index=2 * self.node_index + 2,
                    internal_eps=internal_eps,
                )

        def build(self, input_shape):
            if self.leaf:
                self.leaf_weight = self.add_weight(
                    shape=(1, self.leaf_dims, self.num_trees),
                    trainable=True,
                    name=f"Node-{self.node_index}",
                )

        def call(self, inputs, prob=1.0):
            if self.node_index == 0:
                output = self.dense_layer(inputs)
                output = tf.reshape(
                    output,
                    shape=(tf.shape(output)[0], self.num_trees, self.max_split_nodes),
                )
            else:
                output = inputs

            if not self.leaf:
                current_prob = tf.keras.backend.clip(
                    self.activation(tf.reduce_sum(output * self.masking, axis=-1)),
                    self.internal_eps,
                    1 - self.internal_eps,
                )
                return (
                    self.left_child(output, current_prob * prob)
                    + self.right_child(output, (1 - current_prob) * prob)
                )

            return tf.reduce_sum(tf.expand_dims(prob, axis=1) * self.leaf_weight, axis=2)

    lr_schedule = ConstantLearningRate(learning_rate)
    max_split_nodes = num_trees * (2 ** max_depth - 1)
    tree_layer = TreeEnsembleWithGroupSparsity(
        num_trees=num_trees,
        max_depth=max_depth,
        leaf_dims=1,
        kernel_regularizer=tf.keras.regularizers.L2(kernel_l2 / max_split_nodes),
        kernel_constraint=ProximalGroupL0(
            lr=lr_schedule,
            lam=kernel_constraint / n_features,
            use_annealing=anneal,
            temperature=temperature,
        ),
    )

    inputs = tf.keras.layers.Input(shape=(n_features,))
    model = tf.keras.Model(inputs=inputs, outputs=tree_layer(inputs))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    if validation_split > 0 and patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )

    weights = tree_layer.dense_layer.get_weights()[0]
    scores = np.linalg.norm(weights, axis=1)
    selected = scores > 0.0
    return scores, selected
