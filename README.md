# MaskRCNN-UNet
# This work is an extension of MaskRCNN.
# This work has not be completely yet. There are some errors in it. I post it out because I want some help to make it
# work. 
# When I run train_shapes.py at "samples/shapes/shapes.py", errors like this:
'''Traceback (most recent call last):
  File "/media/zlyx/新加卷/work6/code/MaskRCNN + Unet/MaskRCNN-Unet/samples/shapes/train_shapes.py", line 274, in <module>
    layers='heads')
  File "/media/zlyx/新加卷/work6/code/MaskRCNN + Unet/MaskRCNN-Unet/mrcnn/modelxmask.py", line 2844, in train
    use_multiprocessing=True,
  File "/usr/local/lib/python3.5/dist-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.5/dist-packages/keras/engine/training.py", line 1413, in fit_generator
    initial_epoch=initial_epoch)
  File "/usr/local/lib/python3.5/dist-packages/keras/engine/training_generator.py", line 214, in fit_generator
    class_weight=class_weight)
  File "/usr/local/lib/python3.5/dist-packages/keras/engine/training.py", line 1213, in train_on_batch
    outputs = self.train_function(ins)
  File "/usr/local/lib/python3.5/dist-packages/keras/backend/tensorflow_backend.py", line 2715, in __call__
    return self._call(inputs)
  File "/usr/local/lib/python3.5/dist-packages/keras/backend/tensorflow_backend.py", line 2675, in _call
    fetched = self._callable_fn(*array_vals)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py", line 1439, in __call__
    run_metadata_ptr)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/errors_impl.py", line 528, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.InvalidArgumentError: Expected begin, end, and strides to be 1D equal size tensors, but got shapes [2,1], [2,1], and [2] instead.
	 [[{{node DetectSmaskBbox/map/while/strided_slice_1}}]]'''
   # But I cannot understand this error, can someone help me? Thanks in advance.
