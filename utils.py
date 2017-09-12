import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import numpy as np
import skimage
import tensorflow as tf
import pandas as pd
import os
import model_RGB

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


def extract_video_features(video_path, num_frames = 80, vgg16_model='./vgg16.tfmodel'):
    print "Extracting video features for: " + os.path.basename(video_path)
    # Load tensorflow VGG16 model and setup computation graph
    with open(vgg16_model, mode='rb') as f:
      fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={ "images": images })
    graph = tf.get_default_graph()

    # Read video file
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        pass

    #extract frames from video
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

    # select num_frames from frame_list if frame_cout > num_frames
    if frame_count > num_frames:
        frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
        frame_list = frame_list[frame_indices]

    # crop/resize each frame
    #cropped_frame_list is a list of frames, where each frame is a height*width*3 ndarray
    cropped_frame_list = np.asarray(map(lambda x: preprocess_frame(x), frame_list))

    # extract fc7 features from VGG16 model for each frame
    # feats.shape = (num_frames, 4096)
    with tf.Session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
      video_feat = sess.run(fc7_tensor, feed_dict={images: cropped_frame_list})

    return video_feat

def get_caption(video_feat, model_path='./models/model-910'):
    print "Generating caption ..."
    #video_feat_path = os.path.join('./temp_RGB_feats', '8e0yXMa708Y_24_33.avi.npy')
    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())
    bias_init_vector = np.load('./data/bias_init_vector.npy')

    # lstm parameters
    dim_image = 4096
    dim_hidden= 512
    n_video_lstm_step = 80
    n_caption_lstm_step = 20
    n_frame_step = 80
    batch_size = 50

    #setup lstm encoder-decoer with attention model
    model = model_RGB.Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    #restore lstm model parameters
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    video_feat = video_feat[None,...]

    if video_feat.shape[1] == n_frame_step:
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

    # run model and obatin the embeded words (indices)
    generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})

    # convert indices to words
    generated_words = ixtoword[generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    generated_sentence = generated_sentence.replace('<bos> ', '')
    generated_sentence = generated_sentence.replace(' <eos>', '')
    print generated_sentence,'\n'

    return generated_sentence

if __name__ == '__main__':
    video_path='./temp_videos/_0nX-El-ySo_83_93.avi'
    video_feat = extract_video_features(video_path)
    get_caption(video_feat)
