#Task5: Neural Style Transfer

'''Problem Statement: Apply the artistic style of one image (e.g., a famous painting) to the content of another image using neural style transfer. '''

# Install TensorFlow (if not already installed)
!pip install tensorflow

# Import necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import files

# Function to load and preprocess images
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Option 1: Upload content image
print("Upload your content image:")
uploaded_content = files.upload()
content_path = next(iter(uploaded_content))

# Option 2: Upload style image
print("Upload your style image:")
uploaded_style = files.upload()
style_path = next(iter(uploaded_style))

# Display uploaded images
content_image = load_img(content_path)
style_image = load_img(style_path)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# Load the pre-trained VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Define content and style layers for feature extraction
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Function to extract feature maps from VGG19 model
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

# Instantiate the VGG19 model with selected layers
style_extractor = vgg_layers(style_layers)
content_extractor = vgg_layers(content_layers)

# Function to compute Gram matrix for style representation
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)

# Function to compute style and content loss
def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

# Custom model to extract style and content features
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {self.content_layers[i]: content_outputs[i] 
                        for i in range(len(self.content_layers))}
        
        style_dict = {self.style_layers[i]: style_outputs[i]
                      for i in range(len(self.style_layers))}
        
        return {'content': content_dict, 'style': style_dict}

# Instantiate the style-content model
extractor = StyleContentModel(style_layers, content_layers)

# Compute style and content feature maps for targets
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Define image variable to optimize
image = tf.Variable(content_image)

# Optimizer and training step function
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image):
    style_weight = 1e-2  # Define style weight here
    content_weight = 1e4  # Define content weight here
    
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))

# Training loop parameters
epochs = 10
steps_per_epoch = 100

# Start time measurement
import time
start = time.time()

# Training loop
for n in range(epochs):
    for m in range(steps_per_epoch):
        train_step(image)
        print('.', end='')
    imshow(image.read_value())
    plt.title("Train step: {}".format(n * steps_per_epoch + m + 1))
    plt.show()

# End time measurement and display total time
end = time.time()
print("Total time: {:.1f} seconds".format(end-start))
