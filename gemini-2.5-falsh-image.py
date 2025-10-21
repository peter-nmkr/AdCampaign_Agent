from google import genai
import os
import base64
import dotenv
dotenv.load_dotenv()

# Load logo image as base64
logo_path = "./Testing_Data/logo.png"
logo_base64 = None
logo_mime = "image/png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")
else:
    print(f"Logo image not found at path: {logo_path}")

#import format image as reference for the image generation aspect ratio
format_path = "./Testing_Data/Format_images/728×90_Top-of-page_banners.png"
format_image_base64 = None
format_image_mime = "image/png"
if os.path.exists(format_path):
    with open(format_path, "rb") as f:
        format_image_base64 = base64.b64encode(f.read()).decode("utf-8")
else:
    print(f"Format image not found at path: {format_path}")

# Assemble prompt
resolution = "728x90"
width, height = map(int, resolution.split('x'))
full_prompt = (
    "A stylized illustration of a bookshelf overflowing with books, each with uniquely designed spines in orange, black, and light blue. The image is framed with abstract geometric shapes in the brand's color palette, adding a modern touch. The mood is one of abundance and choice, highlighting Bücherlurch as a haven for book lovers."
    "Headline: Finde dein nächstes Lieblingsbuch!. Subtext: Unsere Regale sind gefüllt mit einzigartigen Geschichten. Call to Action: Entdecken Sie mehr"
    #f", {width}x{height}, aspect ratio"
)


# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-image"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="use the following logo in the image generation"
                ),
                types.Part.from_bytes(
                    data=base64.b64decode(logo_base64),
                    mime_type=(logo_mime or "image/png")
                ),
                types.Part.from_text(
                    text=full_prompt
                )
                #Enable to guid image format
                #types.Part.from_bytes(
                #    data=base64.b64decode(format_image_base64),
                #    mime_type=(format_image_mime or "image/png")
                #)
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            file_name = f"output_gemini_test{file_index}"
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            print(chunk.text)

if __name__ == "__main__":
    generate()
