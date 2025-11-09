from google import genai
from google.genai import types
import mimetypes
import os
import base64
import dotenv
import sys

dotenv.load_dotenv()

full_prompt = ()


# Helper function to save binary files
def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def remove_text(input_image_path: str, output_image_path: str):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-image"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=open(input_image_path, "rb").read(), mime_type=("image/png")
                ),
                types.Part.from_text(
                    text="Remove all text from this image while preserving the background and other elements."
                ),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
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
        if (
            chunk.candidates[0].content.parts[0].inline_data
            and chunk.candidates[0].content.parts[0].inline_data.data
        ):
            file_name = output_image_path
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            save_binary_file(file_name, data_buffer)
        else:
            print(chunk.text)

    return True


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else input("Folder path: ").strip()
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}")
        sys.exit(1)

    index = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        mime, _ = mimetypes.guess_type(path)
        if mime and mime.startswith("image"):
            print(f"Processing: {path}")
            remove_text(path, f"Text_removal_output/{index}")
            index += 1
