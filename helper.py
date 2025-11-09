import base64
import math
from io import BytesIO
from PIL import Image, ImageText, ImageFont, ImageDraw


def extend_image_to_square(base64_str, fill_color=(255, 255, 255)):
    # Decode base64 to image
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGBA")

    # Determine new square size
    max_side = max(image.size)

    # Create new square image with fill color
    new_image = Image.new("RGBA", (max_side, max_side), fill_color + (255,))

    # Center the original image
    x = (max_side - image.width) // 2
    y = (max_side - image.height) // 2
    new_image.paste(image, (x, y), image)

    # Convert back to base64
    buffered = BytesIO()
    new_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def mask_fill_text_boxes(image_path, image_clean_path, rects, texts):
    img = Image.open(image_path)

    mask = Image.new("L", img.size, 255)
    mask_draw = ImageDraw.Draw(mask)

    for rect in rects:
        mask_draw.rectangle(rect, fill=0)

    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    text_colors = []
    for i, rect in enumerate(rects):
        # replacing the cta box with a solid color obtained form the outer edge of the box
        edge_rect = [rect[0], rect[1] - 2, rect[2], rect[1] - 1]
        img_edge = img.crop(edge_rect)
        box_right = max(rect[0], min(rect[2], rect[0]+50))
        box_bottom = max(rect[1], min(rect[3], rect[1]+50))
        img_text = img.crop([rect[0], rect[1], box_right, box_bottom])
        print("box colors")
        box_colors = group_similar_colors(img_text, threshold = 120)
        print("edge_colors")
        edge_colors = group_similar_colors(img_edge, threshold = 8)
        if len(edge_colors) > 1:
            print(f"The text {texts[i]} is not on a solid background")
            img_clean = Image.open(image_clean_path)
            img_clean = img_clean.crop([rect[0], rect[1], box_right, box_bottom])
            print("clean image colors:", end="")
            clean_colors = group_similar_colors(img_clean, threshold=120)
            # finding the color that is NOT in the clean image
            for c in (col["avg_color"] for col in box_colors):
                found = False
                for cc in (col["avg_color"] for col in clean_colors):
                    if color_distance(cc, c) < 15:
                        found = True
                        break
                if found == False:
                    text_color = c
                    break
                else:
                    text_color = (0,0,0)
            print("Text color:", end="")
            print_colored_blocks([text_color])
        else:
            text_color = None
            d.rectangle(rect, fill=edge_colors[0]["avg_color"])
            best_macht = None
            best_dist = 0
            for color in (col["avg_color"] for col in box_colors):
                dist = color_distance(color, edge_colors[0]["avg_color"])
                print("color_distance", dist)
                if dist > best_dist:
                    best_dist = dist
                    best_macht = color
            text_color = best_macht
            print(f"Text: {texts[i]} has a solid background color:", end="")
            print_colored_blocks([best_macht])
        text = texts[i]
        max_width = rect[2] - rect[0]
        max_height = rect[3] - rect[1]
        font_path = "./fonts/Ubuntu/Ubuntu-Bold.ttf"
        lines, font_size = fill_text_box(text, max_width, max_height, font_path)
        font = ImageFont.truetype(font_path, font_size)
        wrapped_text = "\n".join(lines)
        d.multiline_text(rect[:2], wrapped_text, font=font, fill=text_color)
        



    return {"image": img, "colors": text_colors}


def wrap_text(text, font, max_width):
    """
    Wraps text to fit within the max_width.
    """
    words = text.split()
    lines = []  # Holds each line in the text box
    current_line = []  # Holds the current line under evaluation.

    for word in words:
        # Check the width of the current line with the new word added
        test_line = " ".join(current_line + [word])
        width = ImageText.Text(test_line, font).get_length()
        if width <= max_width:
            current_line.append(word)
        else:
            # If the line is too wide, finalize the current line and start a new one
            lines.append(" ".join(current_line))
            current_line = [word]

    # Add the last line
    if current_line:
        lines.append(" ".join(current_line))

    return lines


def fill_text_box(text, max_width, max_height, font, show_debug=False):
    im = Image.new("RGBA", (max_width, max_height), (0, 70, 70, 255))
    d = ImageDraw.Draw(im)
    current_font = None
    current_font_size = 10
    current_height = 0
    current_text = text

    while current_height < max_height:
        #print("Starting new cycle with font size:", current_font_size)
        current_font = ImageFont.truetype(font, current_font_size)
        lines = wrap_text(text, current_font, max_width)
        current_text = lines[0]

        for line in lines[1:]:
            current_text = current_text + "\n" + line

        bbox = d.multiline_textbbox((0, 0), current_text, font=current_font)
        current_height = max_height - bbox[3]
        current_width = max_width - bbox[2]
        #print("current_height", current_height)
        words = text.split()
        if len(words) == len(lines) and current_width < 0:
            current_font_size -= 1
            current_font = ImageFont.truetype(font, current_font_size)
            output_lines = wrap_text(text, current_font, max_width)
            break
        if current_height < 0:
            current_font_size -= 1
            current_font = ImageFont.truetype(font, current_font_size)
            output_lines = wrap_text(text, current_font, max_width)
            break
        output_text = current_text
        current_font_size += 1

    # oput image for debug
    if show_debug:
        if font is None:
            current_font = ImageFont.load_default(current_font_size)
        else:
            current_font = ImageFont.truetype(font, current_font_size)
        d.multiline_text((0, 0), output_text, font=current_font, fill=(255, 255, 255))
        im.show()

    return (output_lines, current_font_size)


def color_distance(c1, c2):
    """Compute Euclidean distance between two RGB colors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def print_colored_blocks(colors):
    """
    Prints ASCII 219 (█) in the given RGB colors using ANSI escape codes.

    Args:
        colors (tuple[tuple[int, int, int]]): A tuple of RGB color tuples.
    """
    for rgb in colors:
        r, g, b = rgb
        # ANSI escape code for RGB foreground color
        print(f"\033[38;2;{r};{g};{b}m█\033[0m", end="")
    print()  # Newline at the end


def group_similar_colors(img, threshold=30, max_colors=1000000):
    """
    Groups similar colors in an image based on a threshold and returns
    the averaged representative color and total pixel count for each group.

    Args:
        image_path (str): Path to the image file.
        threshold (int): Maximum color distance (Euclidean) to consider two colors as similar.
        max_colors (int): Maximum number of colors Pillow should return from getcolors().

    Returns:
        list[dict]: Each dict contains:
            {
                'avg_color': (r, g, b),
                'pixel_count': int
            }
    """

    # Open image and ensure RGB mode
    img = img.convert("RGB")
    colors = img.getcolors(max_colors)
    if not colors:
        raise ValueError("Too many colors in the image. Increase max_colors.")

    # Sort colors by frequency (helps consistency)
    colors.sort(reverse=True, key=lambda c: c[0])

    def color_distance(c1, c2):
        """Compute Euclidean distance between two RGB colors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    groups = []

    for count, color in colors:
        found_group = False
        for group in groups:
            # Compare color with the current group's representative average
            if color_distance(color, group["avg_color"]) < threshold:
                group["colors"].append((count, color))
                group["pixel_count"] += count
                # Recompute weighted average color
                total_weight = group["pixel_count"]
                avg = [0, 0, 0]
                for w, col in group["colors"]:
                    for i in range(3):
                        avg[i] += col[i] * w
                group["avg_color"] = tuple(int(v / total_weight) for v in avg)
                found_group = True
                break
        if not found_group:
            # Start a new group
            groups.append(
                {"colors": [(count, color)], "avg_color": color, "pixel_count": count}
            )

    # Simplify final output: keep only avg_color and total count
    grouped_colors = [
        {"avg_color": group["avg_color"], "pixel_count": group["pixel_count"]}
        for group in groups
    ]

    for color in grouped_colors:
        rgb = color["avg_color"]
        r, g, b = rgb
        print(f"\033[38;2;{r};{g};{b}m█\033[0m", color["pixel_count"])

    return grouped_colors


