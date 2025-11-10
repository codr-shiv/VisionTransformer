from PIL import Image

path = "dataset/ImageNetSelected/n02018795_102"
im = Image.open(path + ".JPEG")
im.save(path + ".ppm")

