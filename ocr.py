# import pytesseract
# from PIL import Image

# img = Image.open("../sampleImages/41.jpg")
# data = pytesseract.image_to_data(img, lang="chi_tra")
# print(data)

# import tesserocr
# from tesserocr import PyTessBaseAPI
# from PIL import Image

# # print(tesserocr.tesseract_version())
# print(tesserocr.get_languages())

# with PyTessBaseAPI(lang="chi_tra") as api:  # type: ignore
#     api.SetImageFile("../sampleImages/41.jpg")
#     print(api.GetUTF8Text())

# import os
# from paddleocr import PaddleOCR, draw_ocr

# # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# # to switch the language model in order.
# ocr = PaddleOCR(
#     use_angle_cls=True, lang="ch"
# )  # need to run only once to download and load model into memory
# img_path = "./sampleImages/manhua/1.jpg"
# result = ocr.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)


# # draw result
# from PIL import Image

# result = result[0]
# image = Image.open(img_path).convert("RGB")
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(
#     image,
#     boxes,
#     txts,
#     scores,
#     font_path=os.path.join(r"C:\Windows\Fonts", "simsun.ttc"),
# )
# im_show = Image.fromarray(im_show)
# im_show.show(title="OCR Result")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import argparse


def preprocess_image(img):
    img = (
        img.convert("L")
        .resize([3 * _ for _ in img.size], Image.BICUBIC)
        .point(lambda p: p > 75 and p + 100)
    )
    return img


def drawImage(img, bboxes=[], scalling_factor=1.0, threashold=70.0):
    if img.mode == "L":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font_path = os.path.join(r"C:\Windows\Fonts", "simsun.ttc")
    font = ImageFont.truetype(font_path, size=20)
    for box in bboxes:
        x1, y1, x2, y2, text, conf = (
            box["x1"],
            box["y1"],
            box["x2"],
            box["y2"],
            box["text"],
            box["conf"],
        )
        if conf <= threashold:
            continue
        draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 255, 0), width=2)
        draw.text(
            (x1, y1 - 10),
            text,
            font=font,
            fill=(255, 0, 0),
            stroke_width=0,
        )
        draw.text(
            (x1, y1 - 30),
            f"{conf:.0f}",
            font=font,
            fill=(0, 0, 255),
            stroke_width=1,
        )
    # display(img.resize(int(scalling_factor * s) for s in img.size))
    img = img.resize(int(scalling_factor * s) for s in img.size)
    # img.show(title="OCR Result")
    img.save("result.jpg")


def use_PaddleOCR(
    image_path,
    lang,
    scalling_factor=1.0,
    threashold=70.0,
    do_preprocess=False,
    default_draw=True,
):
    img = Image.open(image_path)
    if do_preprocess:
        img = preprocess_image(img)
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    data = ocr.ocr(np.array(img))
    if default_draw:
        data = data[0]
        img = Image.open(image_path).convert("RGB")
        boxes = [line[0] for line in data]
        txts = [line[1][0] for line in data]
        scores = [line[1][1] for line in data]
        print("\n\nBoxes:\t", boxes)
        print("\n\nTexts:\t", txts)
        print("\n\nScores:\t", scores)
        im_show = draw_ocr(
            img,
            boxes,
            txts,
            scores,
            font_path=os.path.join(r"C:\Windows\Fonts", "simsun.ttc"),
        )
        im_show = Image.fromarray(im_show)
        im_show.save("result.jpg")
    else:
        tmp_text = ""
        tmp_boxes = []
        for box in data[0]:
            tmp_data = {}
            tmp_coords = box[0]
            text, conf = box[1]
            if len(tmp_coords) > 3:
                x1, y1, x2, y2 = (
                    tmp_coords[0][0],
                    tmp_coords[0][1],
                    tmp_coords[2][0],
                    tmp_coords[2][1],
                )
                tmp_data["x1"], tmp_data["y1"], tmp_data["x2"], tmp_data["y2"] = (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            tmp_data["text"] = text
            tmp_data["conf"] = conf * 100
            tmp_boxes.append(tmp_data)
            tmp_text += text + "\n"
        print("\n Extracted Text:\n\n", tmp_text)
        drawImage(
            img,
            bboxes=tmp_boxes,
            scalling_factor=scalling_factor,
            threashold=threashold,
        )


MANGA_IMAGES_DIR = r"C:\Workspace\Learning\Projects\Anuvad\python\sampleImages\manga\shinju_no_necktar\chapter_0085"
MANHUA_IMAGES_DIR = r"C:\Workspace\Learning\Projects\Anuvad\python\sampleImages\manhua\my_harem_grew_so_large\chapter_0001"
manga_images = os.listdir(MANGA_IMAGES_DIR)
manga_images_sorted = sorted(manga_images)
manhua_images = os.listdir(MANHUA_IMAGES_DIR)
manhua_images_sorted = sorted(manhua_images)
print(manga_images, manhua_images)

# print("Manga Images in Sorted Order:\n", manga_images_sorted, "\n\n")
# print("Manhua Images in Sorted Order:\n", manhua_images_sorted, "\n\n")
img_idx = 8
mode = "manga"
ocr_tool = "paddleocr"
lang_codes = {
    "tesseract": {"english": "eng", "japanese": "jpn", "chinese": "chi_sim"},
    "easyocr": {"english": "en", "japanese": "ja", "chinese": "ch_sim"},
    "paddleocr": {"english": "en", "japanese": "japan", "chinese": "ch"},
}
LANG = ""
if mode == "manhua":
    LANG = lang_codes[ocr_tool]["chinese"]
elif mode == "manga":
    LANG = lang_codes[ocr_tool]["japanese"]

image_path = (
    os.path.join(MANHUA_IMAGES_DIR, manhua_images_sorted[img_idx])
    if mode == "manhua"
    else os.path.join(MANGA_IMAGES_DIR, manga_images_sorted[img_idx])
)

use_PaddleOCR(
    image_path,
    lang=LANG,
    threashold=0.0,
    scalling_factor=1.0,
    do_preprocess=False,
    default_draw=True,
)


# def main():
#     # Create the parser
#     parser = argparse.ArgumentParser(description='Process some images for OCR.')

#     # Add the arguments
#     parser.add_argument('-t', '--type', choices=['manga', 'manhua', 'manhwa'], required=True,
#                         help='Type of image (manga, manhua, manhwa)')
#     parser.add_argument('-n', '--name', nargs='+', required=True,
#                         help='Name(s) of the type {manga, manhua, manhwa}')
#     parser.add_argument('-c', '--chapter', nargs='+', required=True,
#                         help='Chapter(s) to OCR')
#     parser.add_argument('-nimgs', '--numImages', type=int, required=True,
#                         help='Number of images')

#     # Parse the arguments
#     args = parser.parse_args()

#     # Process the arguments
#     print(f"Type: {args.type}")
#     print(f"Names: {args.name}")
#     print(f"Chapters: {args.chapter}")
#     print(f"Number of Images: {args.numImages}")

#     # Here you can add the logic to process the images based on the provided arguments
#     # For example, you could iterate over the names and chapters and perform OCR on each
#     # image in the specified chapters, up to the specified number of images.

# if __name__ == '__main__':
#     main()
