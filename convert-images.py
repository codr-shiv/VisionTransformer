from PIL import Image

path = "dataset/ImageNetSelected/n15075141_10340"
im = Image.open(path + ".JPEG")
im.save(path + ".ppm")

