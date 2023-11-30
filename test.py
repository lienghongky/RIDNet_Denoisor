import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob
import matplotlib.pyplot as plt
from ModelClass import RIDNetModel,EAM

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6048)])

        print("Physical GPU:", tf.config.list_physical_devices('GPU'))
    except RuntimeError as e:
        print(e)

def load_image_pair(noisy_dir, clean_dir):
  noisy_imgs = sorted(os.listdir(noisy_dir))
  clean_imgs = sorted(os.listdir(clean_dir))

  for n_img, c_img in zip(noisy_imgs, clean_imgs):
    n_img_path = os.path.join(noisy_dir, n_img)
    c_img_path = os.path.join(clean_dir, c_img)

    n_img = img_to_array(load_img(n_img_path, target_size=(256, 256)))/255.0
    c_img = img_to_array(load_img(c_img_path, target_size=(256, 256)))/255.0

    yield (n_img, c_img)

# Replace with your actual directories
train_noisy_dir = './datasets/train/input'
train_clean_dir = './datasets/train/groundtruth'
test_noisy_dir = './datasets/test/input'
test_clean_dir = './datasets/test/groundtruth'
validation_noisy_dir = './datasets/validate/input'
validation_clean_dir = './datasets/validate/groundtruth'

def create_dataset(noisy_dir, clean_dir):
  dataset = tf.data.Dataset.from_generator(
    load_image_pair,
    args=[noisy_dir, clean_dir],
    output_signature=(
      tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
      tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
    )
  )
  return dataset.batch(8).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_noisy_dir, train_clean_dir)
test_dataset = create_dataset(test_noisy_dir, test_clean_dir)
validation_dataset = create_dataset(validation_noisy_dir, validation_clean_dir)

RIDNet = RIDNetModel # Model(input,out)
RIDNet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanSquaredError())
# Load the model
model_name = 'models/RIDNet.h5'
model_name = 'model_checkpoints/weights.03-0.01.keras'
RIDNet = tf.keras.models.load_model(model_name,custom_objects={'EAM':EAM})


# Create a directory to save the images
save_dir = './outputs'
os.makedirs(save_dir, exist_ok=True)


# Get the file paths of the images in the validation directory
validation_image_paths = glob.glob(os.path.join(validation_noisy_dir, '*.png'))

for i, image_path in enumerate(validation_image_paths):
  if i >= 10:
    break
  print(image_path)
  # Load the noisy image
  noisy_image = c_img = img_to_array(load_img(image_path, target_size=(256, 256)))/255.0
  
  # Get the denoised image using the model
  denoised_image = RIDNet.predict(tf.expand_dims(noisy_image, axis=0))[0]
  # get the color image
  denoised_image = denoised_image * 255.0
  noisy_image = noisy_image * 255.0
  #get original image
  c_img = img_to_array(load_img(image_path.replace("input","groundtruth"), target_size=(256, 256)))
  #calculate psnr, ssim then print
  psnr = tf.image.psnr(denoised_image, c_img, max_val=255)
  ssim = tf.image.ssim(denoised_image, c_img, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
  
  print("PSNR:", psnr.numpy())
  print("SSIM:", ssim.numpy())
  # Calculate the average PSNR and SSIM values for the whole dataset
  # psnr = tf.reduce_mean(psnr)
  # ssim = tf.reduce_mean(ssim)

  # Plot the images
  fig, axes = plt.subplots(1, 3, figsize=(10, 5))
  axes[0].imshow(c_img.astype('uint8'))
  axes[0].set_title('Original Image')
  axes[0].axis('off')
  axes[1].imshow(noisy_image.astype('uint8'))
  axes[1].set_title('Noisy Image')
  axes[1].axis('off')
  axes[2].imshow(denoised_image.astype('uint8'))
  axes[2].set_title('Denoised Image')
  axes[2].axis('off')

  # Calculate PSNR and SSIM
  ps = f"    PSNR: {psnr.numpy():.2f}  |  SSIM: {ssim.numpy():.2f}    "

  # Add PSNR and SSIM values as text to the plot below the images
  axes[2].text(denoised_image.shape[1]/2, denoised_image.shape[0] + 20, ps, color='white', backgroundcolor='black', ha='center', va='bottom', fontsize=10)


  # Save the plot as an image
  image_name = f'denoised_image_{i+1}.png'
  save_path = os.path.join(save_dir, image_name)
  plt.savefig(save_path)

  plt.close(fig)  # Close the figure to release memory



# # Evaluate the model
# result = RIDNet.evaluate(validation_dataset)
# print("Validation Loss:", result)

def calculate_average_pnsr_ssim():

  #Calculate average psnr and ssim over the whole dataset
  # print("Average PSNR:", psnr)
  # print("Average SSIM:", ssim)
    
  # Calculate average PSNR and SSIM values for the whole dataset
  avg_psnr = 0.0
  avg_ssim = 0.0
  count = 0

  for i, image_path in enumerate(validation_image_paths):
    if i >= 1000:
      break
    print(image_path)
    # Load the noisy image
    noisy_image = c_img = img_to_array(load_img(image_path, target_size=(256, 256)))/255.0

    # Get the denoised image using the model
    denoised_image = RIDNet.predict(tf.expand_dims(noisy_image, axis=0))[0]
    # get the color image
    denoised_image = denoised_image * 255.0
    noisy_image = noisy_image * 255.0
    #get original image
    c_img = img_to_array(load_img(image_path.replace("input","groundtruth"), target_size=(256, 256)))
    #calculate psnr, ssim then print
    psnr = tf.image.psnr(denoised_image, c_img, max_val=255)
    ssim = tf.image.ssim(denoised_image, c_img, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    print("PSNR:", psnr.numpy())
    print("SSIM:", ssim.numpy())

    # Accumulate PSNR and SSIM values
    avg_psnr += psnr.numpy()
    avg_ssim += ssim.numpy()
    count += 1

  # Calculate average PSNR and SSIM
  avg_psnr /= count
  avg_ssim /= count

  print("Average PSNR:", avg_psnr)
  print("Average SSIM:", avg_ssim)