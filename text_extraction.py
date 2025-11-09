from docling.document_converter import DocumentConverter
from difflib import SequenceMatcher
import json
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageText
import os
from gemini_text_removal import remove_text


def filter_texts_for_copy(texts, copy_elements):
    matches = []
    for copy_element in copy_elements:
        best_score = 0
        best_match = None
        for e in texts:
            text = e.get("orig")
            similarity = SequenceMatcher(
                None, text, copy_elements[copy_element]
            ).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match = e
        matches.append({"copy_element": copy_element, "text": best_match})
    return matches


# Only for debugging visualization
def render_bboxes_on_image(
    image_path, bboxes, out_path=None, color=(255, 0, 0), width=3, label=True
):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(bboxes):
        l = float(box.get("l", 0))
        t = float(box.get("t", 0))
        r = float(box.get("r", 0))
        b = float(box.get("b", 0))
        origin = str(box.get("coord_origin", "")).upper()

        if origin == "BOTTOMLEFT":
            top = H - t
            bottom = H - b
        else:
            top = t
            bottom = b

        left_i = int(round(min(l, r)))
        right_i = int(round(max(l, r)))
        top_i = int(round(min(top, bottom)))
        bottom_i = int(round(max(top, bottom)))

        draw.rectangle([left_i, top_i, right_i, bottom_i], outline=color, width=width)

        if label:
            text = str(i)
            text_pos = (left_i, max(0, top_i - 10))
            draw.text(text_pos, text, fill=color, font=font)

    if out_path is None:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_bboxes.png"
    img.save(out_path)
    return out_path


def convert_bbox_to_rect(bbox, img):
    print(bbox)
    W, H = img.size

    l = float(bbox.get("l", 0))
    t = float(bbox.get("t", 0))
    r = float(bbox.get("r", 0))
    b = float(bbox.get("b", 0))
    origin = str(bbox.get("coord_origin", "")).upper()

    if origin == "BOTTOMLEFT":
        top = H - t
        bottom = H - b
    else:
        top = t
        bottom = b

    left_i = int(round(min(l, r)))
    right_i = int(round(max(l, r)))
    top_i = int(round(min(top, bottom)))
    bottom_i = int(round(max(top, bottom)))

    rect = [left_i, top_i, right_i, bottom_i]

    return rect


if __name__ == "__main__":
    copy_elements = {
        "Headline": "Finde dein nächstes Lieblingsbuch",
        "Subtext": "Unsere Regale sind gefüllt mit einzigartigen Geschichten",
        "CtA": "Entdecken sie mehr",
    }
    source = "./Testing_Data/wrongSpelling_sample.png"
    source_image = Image.open(source)
    source_image.show()

    converter = DocumentConverter()
    result = converter.convert(source)

    doc = result.document

    doc_dict = doc.export_to_dict()

    texts = doc_dict.get("texts", {})
    matches = filter_texts_for_copy(texts, copy_elements)

    bboxes = []
    for match in matches:
        bbox = match["text"]["prov"][0]["bbox"]
        bboxes.append(bbox)

    # bboxes_image = render_bboxes_on_image(source, bboxes)
    # PIL.Image.open(bboxes_image).show()

    print(matches)

    cta_bbox = bboxes[2]
    print(cta_bbox)

    # converting docling boundig boxes to PIL rectangles
    # so they work with PILs coordinate system
    text_rects = []
    for bbox in bboxes:
        rect = convert_bbox_to_rect(bbox, source_image)
        text_rects.append(rect)

    img_masked = mask_fill_text_boxes(source, text_rects)
    img_masked.save("./Testing_Output/masked_text.png")
    img_masked.show()

    out_path = "./Testing_Output/text_removed"
    remove_text(source, out_path)
    image_text_removed = Image.open(f"{out_path}.png")
    image_text_removed.show()
    mask = img_masked.getchannel("A")

    composite = Image.composite(img_masked, image_text_removed, mask)

    composite.save("./Testing_Output/text_removed_composite.png")
    composite.show()

    from helper import fill_text_box

    text_composite = composite.copy()

    d_text = ImageDraw.Draw(text_composite)

    for i, rect in enumerate(text_rects):
        text = list(copy_elements.values())[i]
        text_area_im = composite.crop(rect)
        max_width = rect[2] - rect[0]
        max_height = rect[3] - rect[1]
        lines, font_size = fill_text_box(text, max_width, max_height)
        wrapped_text = "\n".join(lines)
        font = ImageFont.truetype("./fonts/Ubuntu/Ubuntu-Bold.ttf", font_size)
        text_el = ImageText.Text(wrapped_text, font)
        d_text.multiline_text(rect[:2], wrapped_text, font=font, fill=(0, 0, 0))

    text_composite.show()
