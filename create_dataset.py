# Suppose the images are labeled correctly i.e. for train images  as A1.jpg,....,Z3000.jpg
# and for test images as A_test.jpg,......,Z_test.jpg
# Suppose these images are in different external directories,
# and we need to create folders: A,...,Z for respective images

import os
import shutil

# create 26 directories for images
# for a in range(65,90+1):
#     print(chr(a))

# first create 26 directories inside D:\ASL using Pytorch\Input\asl_alphabets\train
# for i in range(65,90+1):
#     path = r"D:\ASL using Pytorch\Input\asl_alphabets\train\\" + chr(i)
#     if not os.path.exists(path):
#         os.makedirs(path)

# Now we move images from external directory into the train dataset directory as:
# for a in range(65, 90+1):
#     path1 = r"D:\ASL using Pytorch\Input\asl_alphabets\train\\" + chr(a)
#     for i in range(1, 3000+1):
#         external_train_path = r"D:\ASL using Pytorch\Input\asl_alphabets\train\\" + chr(a)+str(i)+r".jpg"
#         shutil.move(external_train_path, path1)


# then create 26 directories inside D:\ASL using Pytorch\Input\asl_alphabets\test

# for i in range(65,90+1):
#     path = r"D:\ASL using Pytorch\Input\asl_alphabets\test\\" + chr(i)
#     if not os.path.exists(path):
#         os.makedirs(path)

# Now we move images from external directory into the test dataset directory as:
# for a in range(65,90+1):
#     path2 = r"D:\ASL using Pytorch\Input\asl_alphabets\test\\" + chr(a)
#     external_test_path = r"D:\ASL using Pytorch\Input\asl_alphabets\test\\" + chr(a) + r"_test.jpg"
#     shutil.move(external_test_path, path2)
