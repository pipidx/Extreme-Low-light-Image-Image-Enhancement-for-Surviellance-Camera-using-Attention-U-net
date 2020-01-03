
from PIL import Image
import glob, os

input_dir = './dataset/'
result_dir = './dataset/'

in_files = glob.glob(input_dir + '*.png')
for k in range(len(in_files)):
    in_path = in_files[k]
    in_fn = os.path.basename(in_path)
    file_name = in_fn = os.path.splitext(in_fn)[0]

    # print(in_fn,(len(in_files)))
    img = Image.open(in_path)
    img.save(result_dir + file_name + '.tiff')
    # img.show()

