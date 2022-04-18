from PIL import Image, ImageChops
image1 = Image.open('stylish1.jpg')
image2 = Image.open('stylish2.jpg')
differenceimage = ImageChops.difference(image1,image2)
differenceimage.show()