import argparse
import glob
import random
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures
import functools
from typing import Callable, Any
import os
import numpy as np
import traceback


# global executor for use in the decorator
# executor = concurrent.futures.ProcessPoolExecutor()


# # decorator for parallel execution
# def parallel_executor(executor: concurrent.futures.Executor) -> Callable:
#     def decorator(func: Callable) -> Callable:
#         @functools.wraps(func)
#         def wrapper(*args: Any, **kwargs: Any) -> concurrent.futures.Future:
#             # Submit the function to the executor
#             future = executor.submit(func, *args, **kwargs)
#             return future

#         return wrapper

#     return decorator


def preprocess_image(img: Image.Image, scalling_factor=2.5) -> Image.Image:
    img = (
        img.convert("L")
        .resize([scalling_factor * _ for _ in img.size], Image.BICUBIC)  # type: ignore
        .point(lambda p: p > 75 and p + 100)
    )
    return img


def drawImage(
    img: Image.Image,
    scalling_factor: float,
    threashold: float,
    boxes: list[list[list[float | int]]] = [],
    texts: list[str] = [],
    scores: list[float] = [],
    font_path: str = "",
):
    try:
        if img.mode == "L":
            img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        # font_path = os.path.join(r"C:\Windows\Fonts", "simsun.ttc")
        font = ImageFont.truetype(font_path, size=20)
        if len(boxes) != len(texts) != len(scores):
            print("lengths mismatch!")
            return img

        for box, text, score in zip(boxes, texts, scores):
            x1, y1 = box[0]
            if score <= threashold:
                continue
            draw.polygon(xy=[(x, y) for x, y in box], outline=(255, 0, 0), width=2)
            # draw.rectangle(((x1, y1), (x2, y2)), outline=(0, 255, 0), width=2)
            draw.text(
                (x1, y1 - 10),
                text,
                font=font,
                fill=(255, 0, 0),
                stroke_width=0,
            )
            draw.text(
                (x1, y1 - 30),
                f"{score:.0f}",
                font=font,
                fill=(0, 0, 255),
                stroke_width=1,
            )
        img = img.resize(int(scalling_factor * s) for s in img.size)  # type: ignore
        return img
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        return img


def use_Tesseract(
    image_path: str,
    lang: str,
    font_path,
    scalling_factor: float,
    threashold: float,
    do_preprocess: bool = False,
):
    try:
        import pytesseract

        CONFIG = "--oem 1 --psm 3"

        print(f"Performing ocr on file {os.path.basename(image_path)}...")
        img = Image.open(image_path)
        if do_preprocess:
            img = preprocess_image(img)

        data = pytesseract.image_to_data(img, lang=lang, config=CONFIG).split("\n")[1:]
        if not data:
            return img
        boxes: list[list[list[float | int]]] = []
        texts: list[str] = []
        scores: list[float] = []
        for line in data:
            line_list = line.split("\t")
            if len(line_list) <= 11:
                continue
            x1, y1, w, h = (
                int(line_list[6]),
                int(line_list[7]),
                int(line_list[8]),
                int(line_list[9]),
            )
            boxes.append([[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h]])
            texts.append(line_list[11])
            scores.append(
                float(line_list[10]) if line_list[10] else 0.0,
            )
        img = img.convert("RGB")
        drawn_image = drawImage(
            img,
            scalling_factor,
            threashold,
            boxes=boxes,
            texts=texts,
            scores=scores,
            font_path=font_path,
        )
        return drawn_image

    except Exception as e:
        print(
            f"Exception occurred for Image {os.path.basename(image_path)} in directory {os.path.dirname(image_path)}->\n{e}"
        )
        print(traceback.print_exc())


def use_EasyOCR(
    image_path: str,
    lang: str,
    font_path,
    scalling_factor: float,
    threashold: float,
    do_preprocess: bool = False,
):
    try:
        import easyocr

        print(f"Performing ocr on file {os.path.basename(image_path)}...")
        img = Image.open(image_path)
        if do_preprocess:
            img = preprocess_image(img)
        reader = easyocr.Reader(["en", lang])
        data = reader.readtext(np.array(img) if img.mode == "L" else img)
        if not data:
            return img
        boxes: list[list[list[float | int]]] = [line[0] for line in data]
        texts: list[str] = [line[1] for line in data]
        scores: list[float] = [line[2] for line in data]
        img = img.convert("RGB")
        drawn_image = drawImage(
            img,
            scalling_factor,
            threashold,
            boxes=boxes,
            texts=texts,
            scores=scores,
            font_path=font_path,
        )
        return drawn_image
    except Exception as e:
        print(
            f"Exception occurred for Image {os.path.basename(image_path)} in directory {os.path.dirname(image_path)}->\n{e}"
        )
        print(traceback.print_exc())


def use_PaddleOCR(
    image_path: str,
    lang: str,
    font_path,
    scalling_factor: float,
    threashold: float,
    do_preprocess: bool = False,
    default_draw: bool = False,
):
    try:
        from paddleocr import PaddleOCR, draw_ocr  # type:ignore
        from ppocr.utils.logging import get_logger  # type:ignore
        import logging

        logger = get_logger()
        logger.setLevel(logging.ERROR)

        print(f"Performing ocr on file {os.path.basename(image_path)}...")
        img = Image.open(image_path)
        if do_preprocess:
            img = preprocess_image(img)
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        data = ocr.ocr(np.array(img))[0]
        if not data:
            return img
        boxes: list[list[list[float | int]]] = [line[0] for line in data]
        texts: list[str] = [line[1][0] for line in data]
        scores: list[float] = [line[1][1] for line in data]
        img = img.convert("RGB")
        if default_draw:
            drawn_image = draw_ocr(img, boxes, texts, scores, font_path=font_path)
            drawn_image = Image.fromarray(drawn_image)
        else:
            drawn_image = drawImage(
                img,
                scalling_factor,
                threashold,
                boxes=boxes,
                texts=texts,
                scores=scores,
                font_path=font_path,
            )
        return drawn_image
    except Exception as e:
        print(
            f"Exception occurred for Image {os.path.basename(image_path)} in directory {os.path.dirname(image_path)}->\n{e}"
        )
        print(traceback.print_exc())


# @parallel_executor(executor)
def perform_ocr(
    image_path: str,
    engine: str,
    in_dir: str,
    out_dir: str,
    scalling_factor: float,
    threashold: float,
):
    try:
        font_base_dir = r"C:\Windows\Fonts"
        lang_dict = {
            "japanese": {
                "font": "msgothic.ttc",
                "code_paddleocr": "japan",
                "code_easyocr": "ja",
                "code_tesseract": "jpn",
            },
            "chinese": {
                "font": "simsun.ttc",
                "code_paddleocr": "ch",
                "code_easyocr": "ch_sim",
                "code_tesseract": "chi_sim",
            },
            "english": {
                "font": "",
                "code_paddleocr": "en",
                "code_easyocr": "en",
                "code_tesseract": "eng",
            },
        }
        lang = "japanese" if "manga" in image_path else "chinese"
        image_output_path = modify_path(image_path, in_dir, out_dir, engine)
        create_directory_if_missing(image_output_path)
        if engine == "paddleocr":
            img = use_PaddleOCR(
                image_path,
                lang_dict[lang]["code_paddleocr"],
                f"{font_base_dir}/{lang_dict[lang]['font']}",
                scalling_factor,
                threashold,
            )
        elif engine == "easyocr":
            img = use_EasyOCR(
                image_path,
                lang_dict[lang]["code_easyocr"],
                f"{font_base_dir}/{lang_dict[lang]['font']}",
                scalling_factor,
                threashold,
            )
        elif engine == "tesseract":
            img = use_Tesseract(
                image_path,
                lang_dict[lang]["code_tesseract"],
                f"{font_base_dir}/{lang_dict[lang]['font']}",
                scalling_factor,
                threashold,
            )
        else:
            print(
                "You should provide an ocr engine to be used. supported engines are: tesseract, easyocr and paddleocr"
            )
            return
        if not img:
            print(
                f"can't perform ocr on the image -> {os.path.basename(image_path)} located in {os.path.dirname(image_path)}"
            )
            return
        img.save(image_output_path)
        print(
            f"Successfully saved image {os.path.basename(image_output_path)} on directory {os.path.dirname(image_output_path)}"
        )
    except Exception as e:
        print(
            f"Exception occurred for Image {os.path.basename(image_path)} in directory {os.path.dirname(image_path)}->\n{e}"
        )
        print(traceback.print_exc())


def create_directory_if_missing(file_path: str):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def modify_path(old_path: str, omit_prefix: str, new_prefix: str, engine: str):
    filename = os.path.basename(old_path)
    remaining_path = os.path.dirname(old_path)

    if remaining_path.startswith(omit_prefix):
        remaining_path = os.path.relpath(remaining_path, omit_prefix)
    new_path = os.path.join(new_prefix, remaining_path, f"{engine}_results", filename)
    return new_path


def get_image_paths(patterns: list[str], n: int, source_dir: str) -> list[str]:
    base_dir = source_dir
    image_paths: list[str] | list[None] = []
    for pattern in patterns:
        tmp_paths = glob.iglob(f"{base_dir}/{pattern}")
        for tmp_path in tmp_paths:
            image_paths.append(tmp_path)

    image_paths = (
        random.sample(image_paths, n)
        if n != -1 and n <= len(image_paths)
        else image_paths
    )
    return image_paths


def create_glob_patterns(
    types: list[str], names: list[str], chapters: list[str], files: list[str]
) -> list[str]:
    pattern_list: list[str] = []
    for type in types:
        for name in names:
            for chapter in chapters:
                for file_pattern in files:
                    pattern = f"{type}/{name}/{chapter}/{file_pattern}"
                    pattern_list.append(pattern)
    return pattern_list


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some images for OCR.")

    # Add the --all argument
    parser.add_argument("--all", action="store_true", help="Process all images")

    # Add the other arguments
    parser.add_argument(
        "-t",
        "--types",
        nargs="+",
        choices=["manga", "manhua", "manhwa"],
        help="Type of image (manga, manhua, manhwa)",
    )
    parser.add_argument(
        "-n", "--names", nargs="+", help="Name(s) of the type {manga, manhua, manhwa}"
    )
    parser.add_argument("-c", "--chapters", nargs="+", help="Chapter(s) to OCR")
    parser.add_argument("-f", "--files", nargs="+", help="File(s) to OCR")
    parser.add_argument("-nimgs", "--num_images", type=int, help="Number of images")
    parser.add_argument("-e", "--ocr_engine", type=str, help="OCR engine")
    parser.add_argument("-i", "--in_dir", type=str, help="Base source directory")
    parser.add_argument("-o", "--out_dir", type=str, help="Base output directory")
    parser.add_argument(
        "-scf",
        "--scalling_factor",
        type=float,
        help="Images will be scaled proportional to this factor",
    )
    parser.add_argument(
        "-th",
        "--threashold",
        type=float,
        help="OCR results with confidence score less than this threshold will not show up in the saved image",
    )

    # Parse the arguments
    args = parser.parse_args()
    types: list[str] = args.types
    names: list[str] = args.names
    chapters: list[str] = args.chapters
    files: list[str] = ["*"] if not args.files else args.files
    number_of_images: int = -1 if not args.num_images else args.num_images
    ocr_engine: str = "paddleocr" if not args.ocr_engine else args.ocr_engine
    in_dir: str = "sampleImages" if not args.in_dir else args.in_dir
    out_dir: str = "ocr_results" if not args.out_dir else args.out_dir
    scalling_factor: float = 1.0 if not args.scalling_factor else args.scalling_factor
    threashold: float = 0.0 if not args.threashold else args.threashold
    # Process the arguments
    if args.all:
        print("Processing all images...")
        types, names, chapters = (
            ["*"],
            ["*"],
            ["*"],
        )
    else:
        types = ["*"] if not types else types
        names = ["*"] if not names else names
        chapters = ["*"] if not chapters else chapters

    # create glob patterns from arguments
    patterns = create_glob_patterns(types, names, chapters, files)
    # get image paths based on the glob patterns and arguments
    image_paths = get_image_paths(patterns, number_of_images, in_dir)

    # futures = [
    #     perform_ocr(image_path, ocr_engine, out_dir) for image_path in image_paths
    # ]
    # for future in concurrent.futures.as_completed(futures):  # type:ignore
    #     try:
    #         result = future.result()
    #         print(result)
    #     except Exception as exc:
    #         print(f"An Exception Occurred: {exc}")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # for image_path in image_paths:
        results = list(
            executor.map(
                perform_ocr,
                image_paths,
                [ocr_engine] * len(image_paths),
                [in_dir] * len(image_paths),
                [out_dir] * len(image_paths),
                [scalling_factor] * len(image_paths),
                [threashold] * len(image_paths),
            )
        )
    print("ocr done...")


if __name__ == "__main__":
    main()
    # executor.shutdown()
