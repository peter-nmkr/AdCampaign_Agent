# 1 Find paragraph text and bounds in image
# 2 Map boxes to corresponding text form the original prompt
# 3 Identify font style/color and background color
# 4 Create mask for text bounds, filling over text with solid background color
# 5 Remove text from image using image model
# 6 Create composite image with mask and text-less-image
# 7 Find correct font size to fit text bounds
# 8 Fill text bounds with new text using identified font style/color

from text_extraction import *
import os
import base64
import io
from docling.datamodel.base_models import DocumentStream
from PIL import Image
import .gemini_text_removal
from .helper import mask_fill_text_boxes


def replace_text(image_path, copy_elements):
    image = Image.open(image_path)
    print(copy_elements)
    # 1 fing paragraph text and bounds in image
    converter = DocumentConverter()
    result = converter.convert(image_path)
    doc = result.document
    doc_dict = doc.export_to_dict()
    texts = doc_dict.get("texts", {})

    # 2 find matching text in image
    matches = filter_texts_for_copy(texts, copy_elements)

    # 3 remove text from image using image model
    text_removed_path = f"./Testing_Data/Buecherlurch/Images/Text_removed/{os.path.basename(image_path)}"
    gemini_text_removal.remove_text(image_path, text_removed_path)
    image_text_removed = Image.open(text_removed_path)

    # WIP identify font style/color and background color
    # currently only replacing cta with solid color background

    # getting all bounding boxes for matching text into a list
    bboxes = []
    for match in matches:
        bbox = match["text"]["prov"][0]["bbox"]
        bboxes.append(bbox)

    # converting docling boundig boxes to PIL rectangles
    # so they work with PILs coordinate system
    text_rects = []
    for bbox in bboxes:
        rect = convert_bbox_to_rect(bbox, image)
        text_rects.append(rect)
    print("-------Texts structure------")
    print(matches)
    insert_texts = list(copy_elements.values())
    print("insert_texts", insert_texts)
    mask_info = mask_fill_text_boxes(image_path, text_removed_path, text_rects, insert_texts)
    img_masked = mask_info["image"]

    # 6 create composite image with mask and text-less-image
    mask = img_masked.getchannel("A")

    composite = Image.composite(img_masked, image_text_removed, mask)

    composite_path = f"./Testing_Data/Buecherlurch/Images/composite/{os.path.basename(image_path)}"
    composite.save(composite_path)

    # 7 find correct font size to fit text bounds

    from helper import fill_text_box

    text_composite = composite.copy()

    d_text = ImageDraw.Draw(text_composite)

    #for i, rect in enumerate(text_rects):
    #    text = list(copy_elements.values())[i]
    #    text_area_im = composite.crop(rect)
    #    max_width = rect[2] - rect[0]
    #    max_height = rect[3] - rect[1]
    #    font_path = "./fonts/Ubuntu/Ubuntu-Bold.ttf"
    #    lines, font_size = fill_text_box(text, max_width, max_height, font_path)
    #    font = ImageFont.truetype(font_path, font_size)
    #    wrapped_text = "\n".join(lines)
    #    d_text.multiline_text(rect[:2], wrapped_text, font=font, fill=(0, 0, 0))

    return composite_path

if __name__ == "__main__":
    import json

    # importing sample graphics concepts from json file
    with open("./Testing_Data/Buecherlurch/concepts.json", "r") as f:
        concepts = json.load(f)

    # loop over all images in the givene folder and match them with the concept of the same index
    dir_path = "./Testing_Data/Buecherlurch/Images/"
    dir = os.listdir(dir_path)

    text_image_pairs = []
    print(len(concepts))
    for i, file in enumerate(dir):
        # if i > 0: break
        if os.path.splitext(file)[1] == ".png":
            print("index", i)
            image_path = os.path.join(dir_path, str(i + 1) + ".png")
            copy_elements = {
                "copy_headline": concepts[i]["copy_headline"],
                "copy_subtext": concepts[i]["copy_subtext"],
                "call_to_action": concepts[i]["call_to_action"],
            }
            pair = {"image_path": image_path, "copy_elements": copy_elements}
            text_image_pairs.append(pair)

    for pair in text_image_pairs:
        replace_text(pair["image_path"], pair["copy_elements"])
