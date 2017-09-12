#-*- coding: utf-8 -*-
'''
functions to extract VGG (or any other) features from videos:
For each video:
1. extract frames from video (using Opencv)
2. select n (n=80 in this case) frames
3. rezie and crop the selected frames
4.  For each frames:
    1. feed to the VGG-16 model (providied with pre-trained weight, using Caffe)
    2. obtain the VGG features: 1*4096
5. obtain the VGG features of all n frames and store the result as an array (shape: n*4096)
'''
import sys
import cv2
import os
import numpy as np
import skimage
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,3"  

def preprocess_frame(image, target_height=224, target_width=224):
    #function to resize frames then crop
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_width,target_height))

    elif height < width:
        #cv2.resize(src, dim) , where dim=(width, height)
        #image.shape[0] returns height, image.shape[1] returns width, image.shape[2] reutrns 3 (3 RGB channels)
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_height))
        cropping_length = int((resized_image.shape[1] - target_width) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_width, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_height) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_width, target_height))

def main():
    num_frames = 80
    video_path='/home/zhiyong/final_project/data/MSVD/YouTubeClips'
    video_save_path = './temp_RGB_feats'
    videos = os.listdir(video_path)
    videos = filter(lambda x: x.endswith('avi'), videos)

    #Load tensorflow VGG16 model
    with open("vgg16.tfmodel", mode='rb') as f:
      fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={ "images": images })
    graph = tf.get_default_graph()

    # Processing each video in video_path
    for idx, video in enumerate(videos):
        print idx, video

        if os.path.exists( os.path.join(video_save_path, video) ):
            print "Already processed ... "
            continue

        video_fullpath = os.path.join(video_path, video)
        try:
            cap  = cv2.VideoCapture( video_fullpath )
        except:
            pass

        frame_count = 0
        frame_list = []

        while True:
            #extract frames from the video, where each frame is an array (height*width*3)
            ret, frame = cap.read()
            if ret is False:
                break

            frame_list.append(frame)
            frame_count += 1

        frame_list = np.array(frame_list)

        if frame_count > 80:
            #select 80 frames if frame_cout is >80
            frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]

        cropped_frame_list = np.asarray(map(lambda x: preprocess_frame(x), frame_list))

        #cropped_frame_list is a list of frames, where each frame is a height*width*3 array
        with tf.Session() as sess:
          init = tf.global_variables_initializer()
          sess.run(init)
          fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
          feats = sess.run(fc7_tensor, feed_dict={images: cropped_frame_list})

        save_full_path = os.path.join(video_save_path, video + '.npy')
        np.save(save_full_path, feats)
        print feats.shape

if __name__=="__main__":
    main()
