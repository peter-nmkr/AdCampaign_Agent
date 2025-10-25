from docling.document_converter import DocumentConverter
from difflib import SequenceMatcher
import json
import PIL
from PIL import Image, ImageDraw, ImageFont
import os

source = "Testing_Data/wrongSpelling_sample.png"

converter = DocumentConverter()
result = converter.convert(source)

doc = result.document

doc_dict = doc.export_to_dict()


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


def remove_cta_from_solid_background(image_path, bbox):
    img = Image.open(image_path)
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

    cbox = [left_i, top_i, right_i, bottom_i]
    edge_cbox = [left_i - 2, top_i, left_i - 1, bottom_i]

    img_croped = img.crop(edge_cbox)
    colors = img_croped.getcolors()
    print("colors:")
    print(colors)
    num_colors = 0
    for color in colors:
        if color[0] > num_colors:
            num_colors = color[0]
    print("num_colors:", num_colors)
    if num_colors > 10:
        print("The CtA is not in a solid background")
    draw = ImageDraw.Draw(img)
    print(cbox)
    draw.rectangle(cbox, fill=colors[0][1])
    return img


copy_elements = {
    "Headline": "Finde dein nächstes Lieblingsbuch",
    "Subtext": "Unsere Regale sind gefüllt mit einzigartigen Geschichten",
    "CtA": "Entdecken sie mehr",
}

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
img_no_cta = remove_cta_from_solid_background(source, cta_bbox)
img_no_cta.save("./Testing_Output/cta_removed.png")
img_no_cta.show()
