# import pytesseract
# from PIL import Image

# img = Image.open("../sampleImages/41.jpg")
# data = pytesseract.image_to_data(img, lang="chi_tra")
# print(data)

import tesserocr
from tesserocr import PyTessBaseAPI
from PIL import Image

# print(tesserocr.tesseract_version())
print(tesserocr.get_languages())

with PyTessBaseAPI(lang="chi_tra") as api:  # type: ignore
    api.SetImageFile("../sampleImages/41.jpg")
    print(api.GetUTF8Text())
