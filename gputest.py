import tensorflow as tf

# TensorFlow가 GPU를 인식하는지 확인
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow가 GPU를 인식하고 있습니다.")
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print("Device name:", gpu.name)
else:
    print("TensorFlow가 GPU를 인식하지 못하고 있습니다.")
