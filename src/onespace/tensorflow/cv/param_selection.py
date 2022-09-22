import tensorflow as tf

def learning_rate_decay(lr_scheduler):
    if lr_scheduler == "CosineDecay":
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate =0.01,decay_steps = 1000)
    elif lr_scheduler == "ExponentialDecay":
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate =0.01, decay_steps=10000, decay_rate=0.96
                        )
    elif lr_scheduler == "PolynomialDecay":
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                        initial_learning_rate = 0.01,
                        decay_steps = 10000,
                        end_learning_rate = 0.0001,
                        power=0.5)
    elif lr_scheduler == "InverseTimeDecay":
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                        initial_learning_rate = 0.1, decay_steps = 1.0, decay_rate = 0.5)

    elif lr_scheduler == "CosineDecayRestarts":
        lr_schedule =   tf.keras.optimizers.schedules.CosineDecayRestarts(
                                            initial_learning_rate = 0.1,
                                            first_decay_steps = 1000)
    return lr_schedule

        
def get_optimizer(OPTIMIZER, lr_schedule):
    if OPTIMIZER == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    elif OPTIMIZER == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif OPTIMIZER == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    elif OPTIMIZER == "Adadelta":
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr_schedule)
    elif OPTIMIZER == "Adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    elif OPTIMIZER == "Adamax":
        optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
    elif OPTIMIZER == "Nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr_schedule)
    elif OPTIMIZER == "Ftrl":
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr_schedule)
    return optimizer 