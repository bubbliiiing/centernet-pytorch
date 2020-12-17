from centernet import CenterNet
from PIL import Image

centernet = CenterNet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = centernet.detect_image(image)
        r_image.show()
        