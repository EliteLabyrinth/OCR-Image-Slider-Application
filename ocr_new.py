import argparse
import glob
import random
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures
import functools
from typing import Callable, Any
import os
import numpy as np
from paddleocr import PaddleOCR, draw_ocr  # type:ignore


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
    boxes: list[list[list[float | int]]] = [],
    texts: list[str] = [],
    scores: list[float] = [],
    font_path: str = "",
    image_save_path: str = "",
    scalling_factor=1.0,
    threashold=70.0,
):
    if img.mode == "L":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    # font_path = os.path.join(r"C:\Windows\Fonts", "simsun.ttc")
    font = ImageFont.truetype(font_path, size=20)
    if len(boxes) != len(texts) != len(scores):
        print(
            f"Length mismatch between ocr boxes, texts and scores for image {os.path.basename(image_save_path)}. Couldn't save the image!"
        )
        return
    for box, text, score in zip(boxes, texts, scores):
        x1, y1 = box[0]
        x2, y2 = box[1]
        if score <= threashold:
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
            f"{score:.0f}",
            font=font,
            fill=(0, 0, 255),
            stroke_width=1,
        )
    img = img.resize(int(scalling_factor * s) for s in img.size)  # type: ignore
    img.save(image_save_path)


def use_PaddleOCR(
    image_path: str,
    lang: str,
    scalling_factor: float = 1.0,
    threashold: float = 70.0,
    do_preprocess: bool = False,
    default_draw: bool = True,
):
    img = Image.open(image_path)
    if do_preprocess:
        img = preprocess_image(img)
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    data = ocr.ocr(np.array(img))[0]
    boxes = [line[0] for line in data]
    texts = [line[1][0] for line in data]
    scores = [line[1][1] for line in data]
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


# @parallel_executor(executor)
def perform_ocr(image_path: str, engine: str, out_dir: str):
    print(image_path, engine, out_dir)
    return image_path


def get_image_paths(patterns: list[str], n: int) -> list[str]:
    base_dir = "./sampleImages"
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
    types: list[str], names: list[str], chapters: list[str]
) -> list[str]:
    pattern_list: list[str] = []
    for type in types:
        for name in names:
            for chapter in chapters:
                pattern = f"{type}/{name}/{chapter}/*"
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
    parser.add_argument("-nimgs", "--num_images", type=int, help="Number of images")
    parser.add_argument("-e", "--ocr_engine", type=str, help="OCR engine")
    parser.add_argument("-o", "--out_dir", type=str, help="Base output directory")

    # Parse the arguments
    args = parser.parse_args()
    types: list[str] = args.types
    names: list[str] = args.names
    chapters: list[str] = args.chapters
    number_of_images: int = -1 if not args.num_images else args.num_images
    ocr_engine: str = "paddleocr" if not args.ocr_engine else args.ocr_engine
    out_dir: str = "ocr_results" if not args.out_dir else args.out_dir
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
    patterns = create_glob_patterns(types, names, chapters)
    # get image paths based on the glob patterns and arguments
    image_paths = get_image_paths(patterns, number_of_images)

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
                [out_dir] * len(image_paths),
            )
        )
        print(results)


if __name__ == "__main__":
    main()
    # executor.shutdown()
