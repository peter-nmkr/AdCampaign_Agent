from openai import OpenAI
import dotenv
dotenv.load_dotenv()
client = OpenAI()
import base64
from PIL import Image

image_path = "./Testing_Data/wrongSpelling_sample.png"
logo_path = "./Testing_Data/logo.png"

# 2️⃣ Generate a new version based on the original image
response = client.images.edit(
    model="gpt-image-1",
    prompt="Use the provided logo and replace the text accordingly with the following Copy Elements. Headline: Finde dein nächstes Lieblingsbuch!. Subtext: Unsere Regale sind gefüllt mit einzigartigen Geschichten. Call to Action: Entdecken Sie mehr",
    image=[
        open(image_path, "rb"),
        open(logo_path, "rb")
    ],
    size="auto",
)

image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("output_text_cleanup.png", "wb") as f:
    f.write(image_bytes)
