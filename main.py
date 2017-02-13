from vgg import vgg16
import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
from imagenet_classes import class_names
from scipy.misc import imread, imresize

flags = tf.app.flags
flags.DEFINE_string("input", "laska.png", "Path to input image ['laska.png']")
flags.DEFINE_string("output", "laska_save.png", "Path to input image ['laska_save.png']")
flags.DEFINE_string("layer_name", "pool5", "Layer till which to backpropagate ['pool5']")

FLAGS = flags.FLAGS


def load_image(img_path):
	print("Loading image")
	img = imread(img_path, mode='RGB')
	img = imresize(img, (224, 224))
	# Converting shape from [224,224,3] tp [1,224,224,3]
	x = np.expand_dims(img, axis=0)
	# Converting RGB to BGR for VGG
	x = x[:,:,:,::-1]
	return x, img

def main(_):
	x, img = load_image(FLAGS.input)

	sess = tf.Session()

	print("\nLoading Vgg")
	imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
	vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

	print("\nFeedforwarding")
	prob = sess.run(vgg.probs, feed_dict={vgg.imgs: x})[0]
	preds = (np.argsort(prob)[::-1])[0:5]

	print('\nTop 5 classes are')
	for p in preds:
	    print(class_names[p], prob[p])

	# Target class
	predicted_class = preds[0]
	print('predicted_class',predicted_class)

	# Number of output classes of model being used
	nb_classes = 1000

	req_class = np.array([int(i == predicted_class) for i in range(nb_classes)])
	req_class = np.expand_dims(req_class, 0)
	req_class = req_class.astype(np.float32)

	cam = sess.run(vgg.cam, feed_dict={vgg.imgs: x,
		vgg.req_class: req_class})


	cam = np.maximum(cam, 0) + 1e-5
	cam = cam / np.max(cam)
	cam = resize(cam, (224,224))

	# Converting grayscale to 3-D
	cam3 = np.expand_dims(cam, axis=2)
	cam3 = np.tile(cam3,[1,1,3])

	img = img.astype(float)
	img /= (img.max() + 1e-5)

	# Superimposing the visualization with the image.
	new_img = img+3*cam3
	new_img /= (new_img.max() + 1e-5)

	# Display and save
	io.imshow(new_img)
	plt.show()
	io.imsave(FLAGS.output, new_img)

if __name__ == '__main__':
	tf.app.run()

