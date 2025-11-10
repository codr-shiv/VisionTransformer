# ViT-Tiny Inference Pipeline in Pure C

To run the program, follow the following steps after setting the terminal address to current directory -

### To run the code with currently loaded image -

Step 1 - Run ./build.sh

Step 2 - Run ./main


### To load a different image from preloaded dataset -

Step 1 - Open local directory "/dataset/ImageNetSelected/"

Step 2 - Choose any image file in .ppm format, and set its address in main.c file, LoadImgFromPPM() function call in line 11. If not in .ppm format, goto step 2.1, else ignore.

Step 2.1 - If image is not in .ppm format, set .jpeg image path in convert-images.py line 3.

Step 3 - Run ./build.sh

Step 4 - Run ./main
