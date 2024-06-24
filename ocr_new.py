import argparse
import glob
import random
from PIL import Image


def do_ocr(patterns: list[str], n: int, engine: str):
    base_dir = "./sampleImages"
    image_paths = []
    for pattern in patterns:
        tmp_paths = glob.iglob(f"{base_dir}/{pattern}")
        for tmp_path in tmp_paths:
            image_paths.append(tmp_path)

    image_paths = (
        random.sample(image_paths, n)
        if n != -1 and n <= len(image_paths)
        else image_paths
    )


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

    # Parse the arguments
    args = parser.parse_args()
    types: list[str] = args.types
    names: list[str] = args.names
    chapters: list[str] = args.chapters
    number_of_images: int = args.num_images
    ocr_engine: str = args.ocr_engine
    # Process the arguments
    if args.all:
        print("Processing all images...")
        types, names, chapters, number_of_images, ocr_engine = (
            ["*"],
            ["*"],
            ["*"],
            -1,
            "paddleocr",
        )

        # Add logic to process all images
    else:
        types = ["*"] if not types else types
        names = ["*"] if not names else names
        chapters = ["*"] if not chapters else chapters
        number_of_images = -1 if not number_of_images else number_of_images
        ocr_engine = "peddleocr" if not ocr_engine else ocr_engine

    # create glob patterns from arguments
    patterns = create_glob_patterns(types, names, chapters)
    # perform ocr on the files based on the glob patterns and arguments
    do_ocr(patterns, number_of_images, ocr_engine)


if __name__ == "__main__":
    main()
