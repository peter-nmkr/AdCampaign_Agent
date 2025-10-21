# Install dependencies first (run this in your terminal, not in code)
# pip install docling matplotlib

from docling.document import Document
import matplotlib.pyplot as plt

# --- 1. Load the image ---
image_path = "example_page.png"  # <-- Replace this with your image path
doc = Document.from_file(image_path)

# --- 2. Get the first page layout ---
page = doc.pages[0]
layout = page.layout

# --- 3. Print detected layout elements with bounding boxes ---
print("Detected layout elements:")
for block in layout.blocks:
    x0, y0, x1, y1 = block.bbox
    print(f"{block.category}: bbox=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")

# --- 4. Visualize the layout over the image ---
fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(page.image)  # Display the original image

# Draw bounding boxes for each detected block
for block in layout.blocks:
    x0, y0, x1, y1 = block.bbox
    width, height = x1 - x0, y1 - y0

    # Draw a rectangle for each block
    rect = plt.Rectangle(
        (x0, y0), width, height,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    # Add label
    ax.text(x0, y0 - 5, block.category, color='red', fontsize=10, backgroundcolor='white')

ax.set_title("Docling Layout Detection", fontsize=16)
ax.axis("off")
plt.show()
