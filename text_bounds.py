import pytesseract
from PIL import Image
from PIL import ImageDraw, ImageFont
import sys


def get_text_bounds(image_path, conf_threshold):
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    print(data)
    words = data["word_num"]
    # delete duplicate indexes in words
    unique_words = []
    for i in words:
        if i not in unique_words:
            unique_words.append(i)
    print(unique_words)


def group_text_bounds(bounds):
    # Group bounds based on proximity and similar height
    grouped_bounds = []
    threshold_distance = 20  # pixels
    threshold_height = 0.2  # 20% height difference allowed

    used = [False] * len(bounds)
    for i, b1 in enumerate(bounds):
        if used[i]:
            continue
        group = [b1]
        used[i] = True
        for j, b2 in enumerate(bounds):
            if used[j]:
                continue
            # Check height similarity
            height_diff = abs(b1["height"] - b2["height"]) / max(
                b1["height"], b2["height"]
            )
            # Check proximity (vertical and horizontal)
            dx = abs(b1["left"] - b2["left"])
            dy = abs(b1["top"] - b2["top"])
            if height_diff < threshold_height and (
                dx < threshold_distance or dy < threshold_distance
            ):
                group.append(b2)
                used[j] = True
        grouped_bounds.append(group)
    # Calculate the bounding box for each group
    final_bounds = []
    for group in grouped_bounds:
        left = min(b["left"] for b in group)
        top = min(b["top"] for b in group)
        right = max(b["left"] + b["width"] for b in group)
        bottom = max(b["top"] + b["height"] for b in group)
        width = right - left
        height = bottom - top
        texts = [b["text"] for b in group]
        final_bounds.append(
            {"texts": texts, "left": left, "top": top, "width": width, "height": height}
        )
    return final_bounds


def render_bounds_on_image(image_path, bounds, output_path=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for item in bounds:
        left, top, width, height = (
            item["left"],
            item["top"],
            item["width"],
            item["height"],
        )
        draw.rectangle([left, top, left + width, top + height], outline="red", width=2)

    if output_path:
        image.save(output_path)
    else:
        image.show()


# Example usage:
if __name__ == "__main__":
    cinfidence_threshold = int(sys.argv[1] if len(sys.argv) > 1 else 70)
    image_path = "Testing_Data/wrongSpelling_sample.png"  # Replace with your image path
    get_text_bounds(image_path, cinfidence_threshold)
