
import tensorflow as tf 

def build_strategy(MIXED_PRECISION=True, XLA_ACCELERATE=False):
    try:  # detect TPUs
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  
        strategy = tf.distribute.TPUStrategy(tpu)
        DEVICE = 'TPU'
    except ValueError:  # detect GPUs
        strategy = tf.distribute.get_strategy() 
        DEVICE = 'GPU'
        
    if DEVICE == "GPU":
        physical_devices = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(physical_devices))
        try: 
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            assert tf.config.experimental.get_memory_growth(physical_devices[0])
        except: # Invalid device or cannot modify virtual devices once initialized.
            pass 
        
    if MIXED_PRECISION:
        dtype = 'mixed_bfloat16' if DEVICE == "TPU" else 'mixed_float16'
        tf.keras.mixed_precision.set_global_policy(dtype)
        dtype_model = tf.bfloat16
        print('Mixed precision enabled')
    else:
        dtype_model = tf.float32


    if XLA_ACCELERATE:
        tf.config.optimizer.set_jit(True)
        print('Accelerated Linear Algebra enabled')
        
        
    print('REPLICAS           : ', strategy.num_replicas_in_sync)
    print('TensorFlow Version : ', tf.__version__)
    print('Eager Mode Status  : ', tf.executing_eagerly())
    print('TF Cuda Built Test : ', tf.test.is_built_with_cuda)
    print(
        'TF Device Detected : ', 
        'Running on TPU' if DEVICE == "TPU" else tf.test.gpu_device_name()
    )

    try:
        print('TF System Cuda V.  : ', tf.sysconfig.get_build_info()["cuda_version"])
        print('TF System CudNN V. : ', tf.sysconfig.get_build_info()["cudnn_version"])
    except:
        pass
    
    return strategy, dtype_model