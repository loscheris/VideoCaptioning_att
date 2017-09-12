# VideoCaptioning_att
A video captioning tool using S2VT method and attention mechanism (TensorFlow) 

Dependencies:
Tensorflow 1.1
OpenCV 3.2

To Use:
1. From https://drive.google.com/file/d/0B6lX401WcyJRX2NqS0dDUmlUa00/view?usp=sharing download models/ and vgg16.tfmodel
2. Set the model paths in util.py
3. python video_captioning.py -fp path/to/video


TO Train:
1. Use extract_RGB_feats.py to extract features of training videos 
2. Set the paths and parameters in model_RGB.py
3. Run python train.py
4. Test by running: python test.py
