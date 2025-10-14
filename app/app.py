import os
import dotenv
dotenv.load_dotenv()
from kodosumi.core import ServeAPI
app = ServeAPI()

from jinja2 import Environment, FileSystemLoader
from kodosumi.core import forms as F
from forms import campaign_form_model

# campaign_form_model imported from forms.py

import fastapi
from kodosumi.core import Launch
from kodosumi.core import Tracer
from kodosumi.response import Markdown, HTML
from pathlib import Path

# Initialize Jinja2 environment after Path is available
templates_dir = Path(__file__).parent / "templates"
env = Environment(
    loader=FileSystemLoader(str(templates_dir)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Optional
import ray
import asyncio
from pydantic import BaseModel, Field
import base64


# Pydantic models for structured output (only for graphic concepts)

class GraphicConcept(BaseModel):
    """Individual graphic concept with detailed description"""
    graphic_number: int = Field(description="Graphic number (1-10 or more)")
    description: str = Field(description="Detailed description of the graphic including composition, colors, style, elements, mood - very specific for AI image generation")
    target_platform: str = Field(description="Target platform (e.g., Instagram Feed, Facebook Ad, Google Display, Email Header)")
    resolution: str = Field(description="Image resolution (e.g., 1080x1080, 1200x628, 1920x1080)")
    copy_headline: str = Field(description="Headline text for the graphic")
    copy_subtext: str = Field(description="Supporting text/subtext")
    call_to_action: str = Field(description="Call to action text")


class GraphicConceptsOutput(BaseModel):
    """Collection of graphic concepts with campaign overview"""
    campaign_overview: str = Field(description="Brief overview of the campaign graphics strategy")
    concepts: List[GraphicConcept] = Field(description="List of all graphic concepts (may be more than 10 if multiple resolutions needed)")


# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.8,
    api_key=os.getenv("OPENAI_API_KEY")
)


# Define the state for the graph
class CampaignState(TypedDict):
    # Input fields
    name: str
    industry: str
    address: Optional[str]
    website: str
    uvp: str
    problem: str
    audience: Optional[str]
    goal: str
    language: str
    offer: Optional[str]
    cta: Optional[str]
    channels: Optional[str]
    budget: Optional[str]

    # Generated outputs
    screenshot_base64: Optional[str]
    brand_identity: Optional[str]
    logo_base64: Optional[str]
    logo_mime: Optional[str]
    company_profile: str
    graphic_concepts: Optional[GraphicConceptsOutput]
    generated_images: List[Dict]
    image_copy_replacements: List[Dict]

    tracer: Tracer


# Node 1: Capture Website Screenshot and Extract Brand Identity
async def capture_and_extract_brand_node(state: CampaignState) -> CampaignState:
    tracer = state["tracer"]
    website_url = state["website"]

    await tracer.markdown("# Step 1: Analyzing Website and Extracting Brand Identity")
    await tracer.markdown(f"Capturing screenshot from **{website_url}**...")

    try:
        # Import Playwright
        from playwright.async_api import async_playwright

        # Capture screenshot
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": 1920, "height": 1080})

            # Navigate to website with timeout
            await page.goto(website_url, timeout=30000, wait_until="networkidle")

            # Take full page screenshot
            screenshot_bytes = await page.screenshot(full_page=True, type="png")
            await browser.close()

            # Convert to base64
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')

            await tracer.markdown("✅ Screenshot captured successfully!")

            # Check for uploaded logo
            fs = await tracer.fs()
            input_files = await fs.ls("in")
            logo_base64 = None
            logo_mime = None

            if input_files:
                await tracer.markdown("📎 Logo file detected, including in brand analysis...")
                # Read the first uploaded file (logo)
                async for local_path in fs.download(input_files[0]["path"]):
                    logo_path = (local_path)
                await tracer.markdown(f"Logo path: {logo_path}")
                with open(logo_path, "rb") as f:
                    logo_base64 = base64.b64encode(f.read()).decode('utf-8')
                # Detect image mime from extension (fallback to PNG)
                logo_ext = Path(logo_path).suffix.lower()
                if logo_ext in [".jpg", ".jpeg"]:
                    logo_mime = "image/jpeg"
                elif logo_ext == ".png":
                    logo_mime = "image/png"
                else:
                    logo_mime = "image/png"

            await tracer.markdown("🔍 Analyzing brand identity with GPT-4o Vision...")

            # Extract brand identity using GPT-4o Vision
            vision_llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY")
            )

            # Adjust prompt based on whether logo is available
            if logo_base64:
                vision_prompt = env.get_template("vision_with_logo.j2").render()
            else:
                vision_prompt = env.get_template("vision_without_logo.j2").render()

            # Build content array for vision API
            content = [{"type": "text", "text": vision_prompt}]

            # Add website screenshot
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_base64}"
                }
            })

            # Add logo if available
            if logo_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{logo_mime};base64,{logo_base64}"
                    }
                })

            response = await vision_llm.ainvoke([
                {
                    "role": "user",
                    "content": content
                }
            ])

            brand_identity = response.content

            await tracer.markdown("✅ Brand identity extracted!")
            await tracer.markdown("---")
            await tracer.markdown("## Extracted Brand Identity")
            await tracer.markdown(brand_identity)
            await tracer.markdown("---")

            return {
                **state,
                "screenshot_base64": screenshot_base64,
                "brand_identity": brand_identity,
                    "logo_base64": logo_base64,
                    "logo_mime": logo_mime,
                "tracer": tracer
            }

    except Exception as e:
        await tracer.markdown(f"⚠️ Error capturing screenshot or extracting brand identity: {str(e)}")
        await tracer.markdown("Continuing without brand identity extraction...")

        return {
            **state,
            "screenshot_base64": None,
            "brand_identity": None,
            "logo_base64": None,
            "logo_mime": None,
            "tracer": tracer
        }


# Node 2: Generate Company Profile using template
async def generate_company_profile_node(state: CampaignState) -> CampaignState:
    tracer = state["tracer"]

    await tracer.markdown("# Step 2: Building Company Profile")
    await tracer.markdown(f"Analyzing **{state['name']}** in the {state['industry']} industry...")

    # Render the template with form inputs and brand identity
    prompt = env.get_template("company_profile.j2").render(
        name=state['name'],
        address=state.get('address', 'Not provided'),
        industry=state['industry'],
        uvp=state['uvp'],
        audience=state.get('audience', 'Not specified'),
        problem=state['problem'],
        goal=state['goal'],
        offer=state.get('offer', 'Not specified'),
        cta=state.get('cta', 'Not specified'),
        budget=state.get('budget', 'Not specified'),
        channels=state.get('channels', 'Not specified'),
        brand_identity=state.get('brand_identity', None)
    )

    response = await llm.ainvoke(prompt)
    company_profile = response.content

    await tracer.markdown("## Company Profile Created")
    await tracer.markdown(company_profile)

    return {
        **state,
        "company_profile": company_profile,
        "tracer": tracer
    }


# (Removed) Generate Campaign Plan node


# Node 3: Generate Graphic Concepts (with structured output)
async def generate_graphic_concepts_node(state: CampaignState) -> CampaignState:
    tracer = state["tracer"]

    await tracer.markdown("# Step 3: Generating Graphics Overview")
    await tracer.markdown("Creating detailed specifications for all campaign graphics...")

    # Check if logo is available
    has_logo = state.get('logo_base64') is not None

    prompt = env.get_template("graphic_concepts.j2").render(
        company_profile=state['company_profile'],
        language=state['language'],
        has_logo=has_logo
    )
    
    await tracer.markdown("### Graphic Concepts Prompt")
    await tracer.markdown(prompt)

    structured_llm = llm.with_structured_output(GraphicConceptsOutput)
    graphic_concepts = await structured_llm.ainvoke(prompt)

    await tracer.markdown(f"## Graphics Overview Created")
    await tracer.markdown(f"**Campaign Overview:** {graphic_concepts.campaign_overview}")
    await tracer.markdown(f"\n**Total Graphics to Generate:** {len(graphic_concepts.concepts)}")
    await tracer.markdown("")

    # Display as table
    await tracer.markdown("### Graphics Table")
    await tracer.markdown("")
    await tracer.markdown("| # | Platform | Resolution | Headline | Description |")
    await tracer.markdown("|---|----------|------------|----------|")
    for concept in graphic_concepts.concepts:
        await tracer.markdown(f"| {concept.graphic_number} | {concept.target_platform} | {concept.resolution} | {concept.copy_headline} | {concept.description} |")

    return {
        **state,
        "graphic_concepts": graphic_concepts,
        "tracer": tracer
    }


# Ray remote function for Gemini image generation
@ray.remote
def generate_image_gemini(concept: dict, concept_number: int, logo_base64: Optional[str] = None, logo_mime: Optional[str] = None) -> Dict:
    """Generate an image using Gemini 2.5 Flash Image and overlay logo if provided"""
    try:
        from google import genai
    except ImportError:
        return {
            "graphic_number": concept_number,
            "target_platform": concept.get('target_platform', 'Unknown'),
            "resolution": concept.get('resolution', 'Unknown'),
            "error": "google-genai package not installed",
            "runtime": 0
        }
    
    import datetime
    import os
    import time
    import io

    try:
        t0 = datetime.datetime.now()

        # Assemble prompt
        resolution = concept.get('resolution', '1080x1080')
        try:
            width, height = map(int, resolution.split('x'))
        except:
            width, height = 1080, 1080
        headline = concept.get('copy_headline', '')
        subtext = concept.get('copy_subtext', '')
        cta = concept.get('call_to_action', '')
        full_prompt = (
            f"{concept['description']}; "
            f"Headline: {headline}. Subtext: {subtext}. Call to Action: {cta}. "
            f"{width}:{height} aspect ratio"
        )

        # Initialize Gemini client
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # Generate image with retry logic
        max_retries = 3
        image_bytes = None

        for attempt in range(max_retries):
            try:
                # Build contents for Gemini; always include logo if provided
                if logo_base64:
                    from google.genai import types
                    contents = [
                        types.Part.from_bytes(
                            data=base64.b64decode(logo_base64),
                            mime_type=(logo_mime or "image/png")
                        ),
                        full_prompt
                    ]
                else:
                    contents = [full_prompt]

                response = client.models.generate_content(
                    model="gemini-2.5-flash-image-preview",
                    contents=contents,
                )

                if not response or not response.candidates:
                    raise ValueError("No candidates in response")

                # Extract image data
                image_parts = [
                    part.inline_data.data
                    for part in response.candidates[0].content.parts
                    if part.inline_data and part.inline_data.data
                ]

                if image_parts and len(image_parts[0]) > 0:
                    image_bytes = image_parts[0]
                    break

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        if not image_bytes or len(image_bytes) < 100:
            return {
                "graphic_number": concept_number,
                "target_platform": concept.get('target_platform', 'Unknown'),
                "resolution": resolution,
                "error": "Failed to generate valid image data"
            }

        runtime = datetime.datetime.now() - t0

        return {
            "graphic_number": concept_number,
            "target_platform": concept.get('target_platform', 'Unknown'),
            "resolution": resolution,
            "image_path": None,
            "image_data": base64.b64encode(image_bytes).decode('utf-8'),
            "headline": concept.get('copy_headline', ''),
            "subtext": concept.get('copy_subtext', ''),
            "cta": concept.get('call_to_action', ''),
            "runtime": runtime.total_seconds(),
            "error": None
        }

    except Exception as e:
        return {
            "graphic_number": concept_number,
            "target_platform": concept.get('target_platform', 'Unknown'),
            "resolution": concept.get('resolution', 'Unknown'),
            "error": str(e),
            "runtime": 0
        }


# Node 4: Generate Images in Parallel
async def generate_images_parallel_node(state: CampaignState) -> CampaignState:
    tracer = state["tracer"]
    concepts = state["graphic_concepts"].concepts

    await tracer.markdown("# Step 4: Generating Images with Gemini 2.5 Flash")
    await tracer.markdown(f"Creating {len(concepts)} graphics in parallel...")

    # Get logo from state
    logo_base64 = state.get('logo_base64')
    logo_mime = state.get('logo_mime')
    if logo_base64:
        await tracer.markdown("🎨 Logo will be incorporated into graphics where mentioned in descriptions")

    # Convert concepts to dicts for Ray
    concept_dicts = [
        {
            "graphic_number": c.graphic_number,
            "description": c.description,
            "target_platform": c.target_platform,
            "resolution": c.resolution,
            "copy_headline": c.copy_headline,
            "copy_subtext": c.copy_subtext,
            "call_to_action": c.call_to_action
        }
        for c in concepts
    ]

    # Launch parallel Ray tasks with logo
    futures = [generate_image_gemini.remote(concept, concept['graphic_number'], logo_base64, logo_mime)
               for concept in concept_dicts]

    # Wait for results with progress tracking
    unready = futures.copy()
    completed = 0
    results = []

    while unready:
        ready, unready = ray.wait(unready, num_returns=1, timeout=1)
        if ready:
            result = ray.get(ready[0])
            results.append(result)
            completed += 1
            progress = completed / len(concepts) * 100

            await tracer.markdown(f"## Image {completed}/{len(concepts)} ({progress:.0f}%)")
            await tracer.markdown(f"**Graphic #{result.get('graphic_number')} - {result.get('target_platform')}** ({result.get('resolution', 'N/A')})")

            if result.get('error'):
                await tracer.markdown(f"❌ Error: {result['error']}")
            else:
                await tracer.markdown(f"✅ Generated in {result['runtime']:.2f}s")
                # Display inline image using base64
                if result.get('image_data'):
                    await tracer.html(f'<img src="data:image/png;base64,{result["image_data"]}" style="max-width: 600px; border-radius: 8px; margin: 10px 0;" />')

            await tracer.html("<hr style='margin: 20px 0;' />")

        await asyncio.sleep(0.1)

    # Sort results by graphic number
    results.sort(key=lambda x: x.get('graphic_number', 999))

    await tracer.markdown(f"## ✨ All {len(concepts)} Images Generated!")

    return {
        **state,
        "generated_images": results,
        "tracer": tracer
    }

@ray.remote
def replace_image_copy_openai(headline: str, subtext: str, call_to_action: str, image_base64: str, graphic_number: int) -> dict:
    """
    Calls OpenAI Vision API to replace text on a provided image using the replace_copy.j2 template.
    Returns a dict with result or error.
    """
    import os
    import datetime
    from openai import OpenAI
    import base64
    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path

    try:
        t0 = datetime.datetime.now()
        
        # Prepare OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "graphic_number": graphic_number,
                "error": "OPENAI_API_KEY not set",
                "runtime": 0
            }
        client = OpenAI(api_key=api_key)

        # Initialize Jinja2 environment for template
        templates_dir = Path(__file__).parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Render the prompt template
        prompt = env.get_template("replace_copy.j2").render(
            copy_headline=headline,
            copy_subtext=subtext,
            call_to_action=call_to_action
        )

        # Call OpenAI API (GPT-4o vision)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ],
                },
            ],
            max_tokens=500,
        )

        runtime = (datetime.datetime.now() - t0).total_seconds()
        
        # Note: Vision API analyzes images but doesn't modify them
        # The response contains instructions/analysis, not a modified image
        return {
            "graphic_number": graphic_number,
            "result": response.choices[0].message.content,
            "input_prompt": prompt,
            "original_image_data": image_base64,
            "error": None,
            "runtime": runtime
        }
    except Exception as e:
        return {
            "graphic_number": graphic_number,
            "error": f"OpenAI Vision API error: {e}",
            "runtime": 0
        }

async def replace_image_copy_node(state: CampaignState) -> CampaignState:
    """
    Node to call replace_image_copy_openai for each generated image in the state in parallel.
    Adds the results to 'image_copy_replacements' field in the state.
    """
    tracer = state["tracer"]
    generated_images = state.get("generated_images", [])
    
    await tracer.markdown("# Step 5: Replacing Copy in Images with GPT Vision")
    await tracer.markdown(f"Analyzing {len(generated_images)} graphics for copy replacement...")
    await tracer.markdown("")
    await tracer.markdown("⚠️ **Note:** GPT-4o Vision analyzes images and provides instructions, but doesn't directly modify them.")
    await tracer.markdown("")
    
    if not generated_images:
        await tracer.markdown("❌ No images to process!")
        return {
            **state,
            "image_copy_replacements": [],
            "tracer": tracer
        }

    # Filter out images with errors
    valid_images = [img for img in generated_images if not img.get('error')]
    
    if not valid_images:
        await tracer.markdown("❌ No valid images to process!")
        return {
            **state,
            "image_copy_replacements": [],
            "tracer": tracer
        }

    # Launch parallel Ray tasks
    futures = [
        replace_image_copy_openai.remote(
            img.get('headline', ''),
            img.get('subtext', ''),
            img.get('cta', ''),
            img.get('image_data', ''),
            img['graphic_number']
        )
        for img in valid_images
    ]

    # Wait for results with progress tracking
    unready = futures.copy()
    completed = 0
    results = []

    while unready:
        ready, unready = ray.wait(unready, num_returns=1, timeout=1)
        if ready:
            result = ray.get(ready[0])
            results.append(result)
            completed += 1
            progress = completed / len(valid_images) * 100

            await tracer.markdown(f"## Image {completed}/{len(valid_images)} ({progress:.0f}%)")
            await tracer.markdown(f"**Graphic #{result.get('graphic_number')}**")

            if result.get('error'):
                await tracer.markdown(f"❌ Error: {result['error']}")
            else:
                await tracer.markdown(f"✅ Analyzed in {result['runtime']:.2f}s")
                if result.get('result'):
                    await tracer.markdown("### Copy Replacement Analysis:")
                    await tracer.markdown(result['result'])

            await tracer.html("<hr style='margin: 20px 0;' />")

        await asyncio.sleep(0.1)

    # Sort results by graphic number
    results.sort(key=lambda x: x.get('graphic_number', 999))

    await tracer.markdown(f"## ✨ All {len(valid_images)} Images Analyzed!")

    return {
        **state,
        "image_copy_replacements": results,
        "tracer": tracer
    }





# Build the LangGraph workflow
workflow = StateGraph(CampaignState)
workflow.add_node("capture_and_extract_brand", capture_and_extract_brand_node)
workflow.add_node("company_profile", generate_company_profile_node)
workflow.add_node("graphic_concepts", generate_graphic_concepts_node)
workflow.add_node("generate_images", generate_images_parallel_node)
workflow.add_node("replace_image_copy", replace_image_copy_node)

workflow.add_edge(START, "capture_and_extract_brand")
workflow.add_edge("capture_and_extract_brand", "company_profile")
workflow.add_edge("company_profile", "graphic_concepts")
workflow.add_edge("graphic_concepts", "generate_images")
workflow.add_edge("generate_images", "replace_image_copy")
workflow.add_edge("replace_image_copy", END)

graph = workflow.compile()


# Main runner function
async def runner(inputs: dict, tracer: Tracer):
    await tracer.markdown(f"# 🚀 AI Campaign Generator")
    await tracer.markdown(f"Creating complete campaign for **{inputs.get('name')}**")

    # Handle uploaded logo if any
    fs = await tracer.fs()
    input_files = await fs.ls("in")

    if input_files:
        await tracer.markdown("## 📎 Logo Uploaded:")
        for file in input_files:
            path = file["path"]
            name = Path(path).name
            await tracer.markdown(f"- [{name}](/files/{tracer.fid}/{path})")

    await tracer.html("<hr style='margin: 30px 0;' />")

    # Run the campaign generation graph
    result = await graph.ainvoke({
        "name": inputs.get("name", ""),
        "industry": inputs.get("industry", ""),
        "address": inputs.get("address"),
        "website": inputs.get("website", ""),
        "uvp": inputs.get("uvp", ""),
        "problem": inputs.get("problem", ""),
        "audience": inputs.get("audience"),
        "goal": inputs.get("goal", ""),
        "language": inputs.get("language", ""),
        "offer": inputs.get("offer"),
        "cta": inputs.get("cta"),
        "channels": inputs.get("channels"),
        "budget": inputs.get("budget"),
        "screenshot_base64": None,
        "brand_identity": None,
        "logo_base64": None,
        "logo_mime": None,
        "company_profile": "",
        "graphic_concepts": None,
        "generated_images": [],
        "image_copy_replacements": [],
        "tracer": tracer
    })

    # Build comprehensive HTML output
    html = ["<div style='font-family: system-ui, -apple-system, sans-serif; max-width: 1400px; margin: 0 auto;'>"]

    # Header
    html.append(f"<h1 style='color: #1e40af; border-bottom: 3px solid #3b82f6; padding-bottom: 10px;'>")
    html.append(f"📊 Campaign for {result['name']}</h1>")
    html.append(f"<p style='font-size: 1.2em; color: #64748b; margin: 20px 0;'>{result['industry']}</p>")

    # Company Profile Section
    html.append("<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin: 30px 0;'>")
    html.append("<h2>🏢 Company Profile</h2>")
    profile_html = result['company_profile'].replace('\n', '<br>')
    html.append(f"<div style='line-height: 1.6;'>{profile_html}</div>")
    html.append("</div>")

    # (Removed) Campaign Plan Section

    # Graphics Overview
    html.append("<div style='background: #fefce8; padding: 30px; border-radius: 12px; margin: 30px 0; border-left: 5px solid #eab308;'>")
    html.append("<h2 style='color: #854d0e;'>🎨 Graphics Overview</h2>")
    html.append(f"<p style='color: #713f12;'>{result['graphic_concepts'].campaign_overview}</p>")
    html.append("</div>")

    # Generated Graphics Section
    html.append("<h2 style='color: #1e40af; margin-top: 50px; font-size: 2em;'>📸 Campaign Graphics</h2>")
    html.append(f"<p style='color: #64748b; margin-bottom: 30px;'>{len(result['generated_images'])} graphics generated</p>")

    # Create a lookup dict for copy replacements by graphic number
    replacements_by_number = {
        replacement['graphic_number']: replacement 
        for replacement in result.get('image_copy_replacements', [])
    }

    for img_result in result['generated_images']:
        if img_result.get('error'):
            html.append(f"<div style='background: #fee; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #dc2626;'>")
            html.append(f"<h3 style='color: #991b1b;'>Graphic #{img_result['graphic_number']} - {img_result.get('target_platform', 'Unknown')}</h3>")
            html.append(f"<p style='color: #dc2626;'>Error: {img_result['error']}</p>")
            html.append("</div>")
        else:
            html.append("<div style='background: white; padding: 25px; border-radius: 12px; margin: 30px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>")
            html.append(f"<h3 style='color: #1e40af;'>Graphic #{img_result['graphic_number']}: {img_result['target_platform']}</h3>")
            html.append(f"<p style='color: #64748b; font-size: 0.9em; margin-bottom: 15px;'><strong>Resolution:</strong> {img_result['resolution']}</p>")

            # Display image
            html.append(f'<img src="data:image/png;base64,{img_result["image_data"]}" style="width: 100%; max-width: 800px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" />')

            # Copy elements
            html.append("<div style='background: #f8fafc; padding: 20px; border-radius: 8px; margin-top: 15px;'>")
            html.append(f"<h4 style='color: #334155; margin-top: 0;'>📝 Copy Elements</h4>")
            html.append(f"<p><strong style='color: #475569;'>Headline:</strong> {img_result['headline']}</p>")
            html.append(f"<p><strong style='color: #475569;'>Subtext:</strong> {img_result['subtext']}</p>")
            html.append(f"<p><strong style='color: #475569;'>Call to Action:</strong> {img_result['cta']}</p>")
            html.append("</div>")

            # Add GPT Vision Copy Replacement Analysis
            replacement = replacements_by_number.get(img_result['graphic_number'])
            if replacement and replacement.get('result'):
                html.append("<div style='background: #f0fdf4; padding: 20px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #22c55e;'>")
                html.append(f"<h4 style='color: #166534; margin-top: 0;'>🔄 Copy Replacement Analysis</h4>")
                replacement_html = replacement['result'].replace('\n', '<br>')
                html.append(f"<div style='color: #15803d; line-height: 1.6;'>{replacement_html}</div>")
                html.append("</div>")
            elif replacement and replacement.get('error'):
                html.append("<div style='background: #fef3c7; padding: 15px; border-radius: 8px; margin-top: 15px;'>")
                html.append(f"<p style='color: #92400e; margin: 0;'>⚠️ Copy Analysis Error: {replacement['error']}</p>")
                html.append("</div>")

            html.append(f"<p style='color: #94a3b8; font-size: 0.85em; margin-top: 10px;'>⚡ Generated in {img_result['runtime']:.2f}s</p>")
            html.append("</div>")

    html.append("</div>")

    await fs.close()

    return HTML("\n".join(html))


# Declare HTTP endpoint
@app.enter(
    path="/",
    model=campaign_form_model,
    summary="AI Campaign Generator with Graphics",
    description="Generate complete advertising campaigns with AI-powered graphics using GPT-4 and Gemini",
    tags=["Marketing", "AI", "Campaign", "Graphics"],
    version="1.0.0",
    author="claude@anthropic.com",
    organization="NMKR"
)
async def enter(request: fastapi.Request, inputs: dict):
    return Launch(
        request,
        "app.app:runner",
        inputs=inputs
    )


from ray import serve

@serve.deployment
@serve.ingress(app)
class CampaignGeneratorAgent:
    pass

fast_app = CampaignGeneratorAgent.bind()
