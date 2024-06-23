import requests
import os
import sys
import aiohttp
import asyncio


async def fetch(session, url, save_path, char_to_extract_num):
    filename = os.path.basename(url)
    filename = prepend_zeros(filename, char_to_extract_num)
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(save_path + f"/{filename}", "wb") as f:
                    f.write(content)
                print(f"Successfully downloaded {filename}")
            else:
                print(f"Failed to download {url}, status code: {response.status}")
    except Exception as e:
        print(f"Exception occurred while downloading {url}: {e}")


async def download_images(urls, save_path, char_to_extract_num="."):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url, save_path, char_to_extract_num) for url in urls]
        await asyncio.gather(*tasks)


def prepend_zeros(filename: str, char_to_extract_num: str = ".") -> str:
    numeric_part, rest = filename.split(char_to_extract_num, 1)
    # Convert the numeric part to an integer
    num = int(numeric_part)

    # Format the numeric part to have exactly 6 digits with leading zeros
    new_numeric_part = f"{num:06d}"

    # Construct the new filename
    new_filename = f"{new_numeric_part}-{rest}"
    return new_filename


def download_manhua():
    try:
        save_path = r"./sampleImages/manhua"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_urls = []
        for i in range(1, 51):
            image_url = f"https://s1.baozimh.com/scomic/meizitaiduozhihaofeishengliao-lizhihengzhiyindongman/0/0-8o7y/{i}.jpg"  # i from 1 to 50
            image_urls.append(image_url)
        for i in range(51, 72):
            image_url = f"https://s1-mha1-nlams.baozicdn.com/scomic/meizitaiduozhihaofeishengliao-lizhihengzhiyindongman/0/0-8o7y/{i}.jpg"  # i from 51 to 71
            image_urls.append(image_url)
        asyncio.run(download_images(image_urls, save_path))
    except Exception as e:
        print(e)


def download_manga():
    try:
        chapter_sepcific_path = r""
        save_path = r"./sampleImages/manga" + chapter_sepcific_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_urls = []
        image_names = [
            "1-66228d3b84417.jpg",
            "2-66228d3bf2943.jpg",
            "3-66228d3c5aabf.jpg",
            "4-66228d3cc2d25.jpg",
            "5-66228d3d3c3bb.jpg",
            "6-66228d3e4867b.jpg",
            "7-66228d3ea607f.jpg",
            "8-66228d3f18b74.jpg",
            "9-66228d3f86ce0.jpg",
            "10-66228d3ff178b.jpg",
            "11-66228d4068ea1.jpg",
            "12-66228d40d8238.jpg",
            "13-66228d413cc26.jpg",
            "14-66228d41eb36a.jpg",
            "15-66228d42618f9.jpg",
            "16-66228d4304b9d.jpg",
            "17-66228d43e3baf.jpg",
            "18-66228d446ca28.jpg",
            "19-66228d45515d9.jpg",
            "20-66228d4652066.jpg",
            "21-66228d4775e96.jpg",
            "22-66228d47f2f86.jpg",
            "23-66228d48cbb06.jpg",
            "24-66228d49555f8.jpg",
            "25-66228d49dcbc0.jpg",
            "26-66228d4a57dd5.jpg",
            "27-66228d4ad3bae.jpg",
            "28-66228d4b56a0b.jpg",
            "29-66228d4c2fa21.jpg",
            "30-66228d4d2b01b.jpg",
        ]
        base_image_link = "https://cdn.kumacdn.club/wp-content/uploads/images/s/shinju-no-nectar/chapter-85/"
        for image_name in image_names:
            image_urls.append(base_image_link + image_name)
        asyncio.run(download_images(image_urls, save_path, char_to_extract_num="-"))
    except Exception as e:
        print(e)


if len(sys.argv) < 2:
    download_manhua()
    download_manga()
elif sys.argv[1] == "manga":
    download_manga()
elif sys.argv[1] == "manhua":
    download_manhua()
else:
    print(f"the following argument {sys.argv[1]} is not supported!")
