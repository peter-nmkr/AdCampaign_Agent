import os
import dotenv
dotenv.load_dotenv()
from kodosumi.core import ServeAPI
app = ServeAPI()

from jinja2 import Template
from kodosumi.core import forms as F

# Company profile generation template
company_profile_template = Template("""
    You are hired to advise a business on creating a digital ad and social media campaign consisting exclusively of static images. The following information is provided. Make reasonable assumptions to fill in the missing information as a basis for further planning:

    Business Background
    Name:
    {{name}}
    Address:
    {{address}}
    Business type and industry
    {{industry}}
    Unique value proposition
    {{uvp}}
    Current positioning: Premium, affordable, innovative, local, etc.?
    MISSING
    Target Audience
    {{audience}}
    Demographics: Age, gender, location, income, education
    MISSING
    Psychographics: Interests, values, lifestyle, motivations
    MISSING
    What problem do they solve for their customers?
    {{problem}}
    Goals & Objectives
    Primary campaign goal: Awareness, leads, sales, repeat customers, event attendance, etc.
    {{goal}}
    KPIs: Click-through rate, cost per lead ...
    MISSING
    Timeframe: How long should the campaign run?
    MISSING
    Offer & Messaging
    Core offer: Product/service, price point, special promotion, or bundle.
    {{offer}}
    Key messaging:What should the ad say to resonate with the target audience?
    MISSING
    Call-to-action:What action do you want people to take (buy, sign up, visit, etc.)?
    {{cta}}
    Budget & Resources
    Ad spend: How much are they willing to invest?
    {{budget}}
    Media mix: Social media, Google Ads, print, radio, local sponsorships, etc.
    Missing
    Creative assets: Photos, testimonials, product shots. No videos or pictures of staff or real people or locations A logo matching the CI is already provided
    MISSING
    Internal capacity: Who will manage the campaign (in-house vs. agency)?
    MISSING
    Channels & Distribution
    Digital platforms already in use:
    Google Ads, Meta Ads,Email Marketing
    Owned media: Website, blog, email list.
    {{channels}}
    Tracking & Optimization
    Analytics setup: Google Analytics, Meta Pixel, CRM integration.
    MISSING
    Reporting cadence: Daily, weekly, monthly.
    MISSING
    Optimization plan: A/B testing headlines, creatives, offers, targeting.
    MISSING
    Feedback loop: How learnings will shape future campaigns.
    MISSING

    {% if brand_identity %}
    EXTRACTED BRAND IDENTITY (from website analysis):
    {{brand_identity}}

    IMPORTANT: Use this brand identity information to:
    1. Ensure all campaign visuals match the existing color palette
    2. Maintain consistency with the established typography and design style
    3. Align campaign tone with the brand personality
    4. Incorporate visual patterns and UI/UX elements that match the brand
    5. Make assumptions about positioning and target audience based on the visual brand analysis
    {% endif %}

    Provide a full overview of the combined information, integrating the brand identity analysis into your assumptions and recommendations.
""")

# Campaign form with company profile inputs
campaign_form_model = F.Model(
    F.Markdown("""
    # AI Campaign Generator Agent
    Generate a complete digital advertising campaign with AI-powered image creation.
    """),
    F.InputText(
        name="name",
        label="Business name",
        placeholder="Name of your business",
        required=True
    ),
    F.Select(
        name="industry",
        label="Business type",
        option=[
            F.InputOption("Healthcare", "healthcare"),
            F.InputOption("Professional Services", "professional services"),
            F.InputOption("Construction & Real Estate", "construction and real estate"),
            F.InputOption("Manufacturing", "manufacturing"),
            F.InputOption("Technology", "technology"),
            F.InputOption("Transportation & Logistics", "transportation and logistics"),
            F.InputOption("Education & Training", "education and training"),
            F.InputOption("Finance & Insurance", "finance and insurance"),
            F.InputOption("Retail & E-commerce", "retail and e-commerce"),
            F.InputOption("Hospitality & Tourism", "hospitality and tourism"),
            F.InputOption("Other", "other")
        ]
    ),
    F.InputText(
        name="address",
        label="Address",
        placeholder="123 Main St, City, State, Country",
        required=False
    ),
    F.InputFiles(
        name="logo",
        label="Upload Logo (PNG, JPG)",
        required=False,
        multiple=False,
        directory=False
    ),
    F.InputText(
        name="website",
        label="Website",
        placeholder="https://www.yourwebsite.com",
        required=True
    ),
    F.InputArea(
        name="uvp",
        label="Unique Value Proposition",
        placeholder="What makes your business unique?",
        required=True
    ),
    F.InputArea(
        name="problem",
        label="Problem you solve",
        placeholder="What problem does your business solve for customers?",
        required=True
    ),
    F.InputText(
        name="audience",
        label="Target audience",
        placeholder="Describe your target audience",
        required=False
    ),
    F.InputArea(
        name="goal",
        label="Primary campaign goal",
        placeholder="What is your primary goal for this campaign? (e.g., brand awareness, lead generation, sales)",
        required=True
    ),
    F.InputText(
        name="language",
        label="Output Language",
        placeholder="Enter the language for your campaign (e.g., English, Spanish, French)",
        required=True
    ),
    F.InputArea(
        name="offer",
        label="Core offer",
        placeholder="Describe your core offer: Product/service, price point, special promotion, or bundle.",
        required=False
    ),
    F.InputArea(
        name="cta",
        label="Call to action",
        placeholder="What action do you want customers to take? (e.g., Sign up, Buy now, Learn more)",
        required=False
    ),
    F.InputArea(
        name="channels",
        label="Owned marketing channels",
        placeholder="List any owned marketing channels (e.g., website, social media, email list)",
        required=False
    ),
    F.InputText(
        name="budget",
        label="Campaign budget",
        placeholder="What is your budget for this campaign? (e.g., $5000 monthly)",
        required=False
    ),
    F.Submit("Generate Campaign"),
    F.Cancel("Cancel")
)

import fastapi
from kodosumi.core import Launch
from kodosumi.core import Tracer
from kodosumi.response import Markdown, HTML
from pathlib import Path

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
    company_profile: str
    graphic_concepts: Optional[GraphicConceptsOutput]
    generated_images: List[Dict]

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

            if input_files:
                await tracer.markdown("📎 Logo file detected, including in brand analysis...")
                # Read the first uploaded file (logo)
                logo_path = input_files[0]["path"]
                logo_bytes = await fs.read(logo_path)
                logo_base64 = base64.b64encode(logo_bytes).decode('utf-8')

            await tracer.markdown("🔍 Analyzing brand identity with GPT-4o Vision...")

            # Extract brand identity using GPT-4o Vision
            vision_llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY")
            )

            # Adjust prompt based on whether logo is available
            if logo_base64:
                vision_prompt = """Analyze the provided website screenshot AND logo file to extract comprehensive brand identity (CI) information.

                    Compare both images to understand the complete brand identity.

                    Provide a detailed analysis including:

                    1. **Color Palette**:
                       - Primary colors (with hex codes if identifiable from both website and logo)
                       - Secondary colors
                       - Accent colors
                       - Background/neutral colors
                       - How the logo colors are used throughout the website

                    2. **Typography & Text Style**:
                       - Font style (modern, classic, playful, professional, etc.)
                       - Text hierarchy and sizing
                       - Typography personality
                       - Any text in the logo and its style

                    3. **Logo & Branding Elements**:
                       - Detailed logo analysis (style, shapes, symbolism)
                       - Logo characteristics (minimalist, detailed, icon-based, wordmark, combination mark, etc.)
                       - How the logo is integrated into the website design
                       - Brand symbols or icons derived from the logo

                    4. **Visual Design Style**:
                       - Overall aesthetic (minimalist, bold, elegant, playful, corporate, etc.)
                       - Layout patterns (grid-based, asymmetric, centered, etc.)
                       - Use of whitespace
                       - Image style (photography, illustrations, abstract, etc.)
                       - How design elements echo the logo

                    5. **Brand Tone & Personality**:
                       - Professional, casual, friendly, authoritative, innovative, traditional, etc.
                       - Emotional tone conveyed by design
                       - Target audience implied by design choices
                       - Brand personality expressed through logo and website

                    6. **UI/UX Patterns**:
                       - Button styles
                       - Call-to-action prominence
                       - Navigation style
                       - Visual hierarchy

                    Be specific and detailed. This information will be used to create advertising campaigns that match the brand's existing identity."""
            else:
                vision_prompt = """Analyze this website screenshot and extract comprehensive brand identity (CI) information.

                    Provide a detailed analysis including:
                    
                    1. **Color Palette**:
                       - Primary colors (with hex codes if identifiable)
                       - Secondary colors
                       - Accent colors
                       - Background/neutral colors
                    
                    2. **Typography & Text Style**:
                       - Font style (modern, classic, playful, professional, etc.)
                       - Text hierarchy and sizing
                       - Typography personality
                    
                    3. **Logo & Branding Elements**:
                       - Logo placement and style
                       - Logo characteristics (minimalist, detailed, icon-based, wordmark, etc.)
                       - Brand symbols or icons
                    
                    4. **Visual Design Style**:
                       - Overall aesthetic (minimalist, bold, elegant, playful, corporate, etc.)
                       - Layout patterns (grid-based, asymmetric, centered, etc.)
                       - Use of whitespace
                       - Image style (photography, illustrations, abstract, etc.)
                    
                    5. **Brand Tone & Personality**:
                       - Professional, casual, friendly, authoritative, innovative, traditional, etc.
                       - Emotional tone conveyed by design
                       - Target audience implied by design choices
                    
                    6. **UI/UX Patterns**:
                       - Button styles
                       - Call-to-action prominence
                       - Navigation style
                       - Visual hierarchy
                    
                    Be specific and detailed. This information will be used to create advertising campaigns that match the brand's existing identity."""

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
                # Detect image format from file extension
                logo_ext = Path(input_files[0]["path"]).suffix.lower()
                logo_mime = "image/png" if logo_ext == ".png" else "image/jpeg"

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
            "tracer": tracer
        }


# Node 2: Generate Company Profile using template
async def generate_company_profile_node(state: CampaignState) -> CampaignState:
    tracer = state["tracer"]

    await tracer.markdown("# Step 2: Building Company Profile")
    await tracer.markdown(f"Analyzing **{state['name']}** in the {state['industry']} industry...")

    # Render the template with form inputs and brand identity
    prompt = company_profile_template.render(
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
    logo_instruction = "- A company logo is available and can be incorporated into designs where appropriate. Mention logo placement in descriptions where it makes sense (e.g., 'company logo in top-right corner')." if has_logo else "- No logo available, do not mention logos in descriptions."

    prompt = f"""Based on this company profile, give an overview of all graphics that will make up the final campaign.

COMPANY PROFILE:
{state['company_profile']}

REQUIREMENTS:
- Language for all copy: {state['language']}
- Static images only (no videos, no real people, no real locations)
{logo_instruction}
- Images should work for: Google Ads, Meta Ads, Email Marketing

Give an overview of all graphics that will make up the final campaign. Format it as a structured list including the following information:
- Graphic number
- Detailed description of the graphic (be very specific about composition, colors, style, objects, mood - at least 4-5 sentences for AI image generation. Do NOT include people{', but MAY mention logo placement if appropriate' if has_logo else ''})
- Target platform (e.g., Instagram Feed, Facebook Ad, Google Display Banner, Email Header)
- Resolution (standard sizes: 1080x1080 for square, 1200x628 for Facebook, 1080x1920 for stories, 728x90 for banners, etc.)
- Headline text (in {state['language']})
- Subtext (in {state['language']})
- Call to action (in {state['language']})

If a graphic is required in multiple resolutions, it should be split across multiple rows with the same full image description.

Create at least 10 distinct graphic concepts (may result in more rows if multiple resolutions are needed).

Also provide a brief campaign overview explaining the graphics strategy."""

    structured_llm = llm.with_structured_output(GraphicConceptsOutput)
    graphic_concepts = await structured_llm.ainvoke(prompt)

    await tracer.markdown(f"## Graphics Overview Created")
    await tracer.markdown(f"**Campaign Overview:** {graphic_concepts.campaign_overview}")
    await tracer.markdown(f"\n**Total Graphics to Generate:** {len(graphic_concepts.concepts)}")
    await tracer.markdown("")

    # Display as table
    await tracer.markdown("### Graphics Table")
    await tracer.markdown("| # | Platform | Resolution | Headline |")
    await tracer.markdown("|---|----------|------------|----------|")
    for concept in graphic_concepts.concepts:
        await tracer.markdown(f"| {concept.graphic_number} | {concept.target_platform} | {concept.resolution} | {concept.copy_headline[:50]}... |")

    return {
        **state,
        "graphic_concepts": graphic_concepts,
        "tracer": tracer
    }


# Ray remote function for Gemini image generation
@ray.remote
def generate_image_gemini(concept: dict, concept_number: int, logo_base64: Optional[str] = None) -> Dict:
    """Generate an image using Gemini 2.5 Flash Image and overlay logo if provided"""
    from google import genai
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
                # Check if description mentions logo and logo is available
                description_lower = concept['description'].lower()
                include_logo = logo_base64 and ('logo' in description_lower)

                # Build contents for Gemini
                if include_logo:
                    # Include logo image as reference
                    from google.genai import types
                    contents = [
                        types.Part.from_text(full_prompt),
                        types.Part.from_bytes(
                            data=base64.b64decode(logo_base64),
                            mime_type="image/png"
                        )
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
    futures = [generate_image_gemini.remote(concept, concept['graphic_number'], logo_base64)
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


# Build the LangGraph workflow
workflow = StateGraph(CampaignState)
workflow.add_node("capture_and_extract_brand", capture_and_extract_brand_node)
workflow.add_node("company_profile", generate_company_profile_node)
workflow.add_node("graphic_concepts", generate_graphic_concepts_node)
workflow.add_node("generate_images", generate_images_parallel_node)

workflow.add_edge(START, "capture_and_extract_brand")
workflow.add_edge("capture_and_extract_brand", "company_profile")
workflow.add_edge("company_profile", "graphic_concepts")
workflow.add_edge("graphic_concepts", "generate_images")
workflow.add_edge("generate_images", END)

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
        "company_profile": "",
        "graphic_concepts": None,
        "generated_images": [],
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
        "app:runner",
        inputs=inputs
    )


from ray import serve

@serve.deployment
@serve.ingress(app)
class CampaignGeneratorAgent:
    pass

fast_app = CampaignGeneratorAgent.bind()

import uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "app:fast_app",
        host="0.0.0.0",
        port=8014,
        reload=True
    )
