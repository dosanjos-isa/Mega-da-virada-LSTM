import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detectadas:", gpus)
print("TF version:", tf.__version__)
if gpus:
    print("GPU APTA PARA TREINO")
else:
    print("NENHUMA GPU DETECTADA - rodando em CPU")
