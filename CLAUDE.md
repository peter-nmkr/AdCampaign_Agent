# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **AI Campaign Generator Agent** that creates complete digital advertising campaigns with AI-powered static image generation. It uses a LangGraph workflow to orchestrate GPT-4o for strategic planning and Gemini 2.0 Flash for image generation.

**Tech Stack:**
- **kodosumi**: Web framework for agent-based applications with form UI and streaming
- **LangGraph**: Workflow orchestration with state machine
- **Ray Serve**: Parallel image generation and deployment
- **GPT-4o (OpenAI)**: Company profile and campaign strategy generation
- **Gemini 2.0 Flash (Google)**: AI image generation
- **Jinja2**: Template rendering for prompts

## Commands

### Running the Application

```bash
# Development server with hot reload (port 8014)
python app.py

# Production deployment with Ray Serve (port 8000)
serve run serve_config.yaml

# Visualize the LangGraph workflow
python visualize_graph.py
```

### Environment Setup

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
```

## Architecture

### LangGraph Workflow (4 Sequential Nodes)

The application uses a **linear state machine** defined in `app.py:543-556`:

```
START → company_profile → campaign_plan → graphic_concepts → generate_images → END
```

**State Type:** `CampaignState` (TypedDict) carries all data through the workflow, including a `tracer` object for streaming UI updates.

#### Node 1: `generate_company_profile_node` (app.py:249)
- Renders Jinja2 template (`company_profile_template`) with user inputs
- Calls GPT-4o to fill missing information with reasonable assumptions
- Outputs: Comprehensive company profile with positioning, demographics, psychographics

#### Node 2: `generate_campaign_plan_node` (app.py:284)
- Takes company profile from Node 1
- Generates strategic campaign plan including: campaign name, KPIs, messaging pillars, visual theme, color palette, ad formats, audience segments
- All strategy is tailored for **static images only** (no video)

#### Node 3: `generate_graphic_concepts_node` (app.py:327)
- Uses **structured output** with Pydantic models (`GraphicConceptsOutput`, `GraphicConcept`)
- Generates 10+ graphic specifications with detailed AI image generation prompts
- Each concept includes: graphic number, platform, resolution, headline/subtext/CTA (in user's language), detailed visual description (composition, colors, style, objects, mood)
- Returns structured data, not free-form text

#### Node 4: `generate_images_parallel_node` (app.py:477)
- **Ray parallel execution**: Spawns Ray remote tasks for each graphic concept
- Each task calls `generate_image_gemini` (app.py:386) using Gemini 2.0 Flash Image Preview
- Implements retry logic (3 attempts with exponential backoff)
- Streams progress updates using `ray.wait()` as images complete
- Returns base64-encoded images for inline display

### Form and UI System

**Form Definition:** `campaign_form_model` (app.py:75-178) uses kodosumi's form DSL:
- `F.InputText`, `F.InputArea`, `F.Select`, `F.InputFiles` for user inputs
- 11 fields covering business profile, campaign goals, budget
- Industry dropdown with 11 preset options

**Entry Point:** `@app.enter()` decorator (app.py:663) registers the form at "/" route and launches the workflow via `Launch(request, "app:runner", inputs=inputs)`

**Runner Function:** `runner()` (app.py:559) orchestrates:
1. Displays uploaded logo files (if any) from `tracer.fs().ls("in")`
2. Invokes the compiled LangGraph with form inputs
3. Builds comprehensive HTML output with all campaign assets
4. Returns `HTML()` response with styled campaign report

### Streaming and Progress Tracking

The `tracer: Tracer` object (passed through state) enables real-time UI updates:
- `await tracer.markdown()` - Stream markdown content
- `await tracer.html()` - Stream HTML content
- `await tracer.fs()` - Access file system for uploads

Progress is shown during parallel image generation by calling `tracer.markdown()` after each Ray task completes (app.py:515-526).

### Ray Deployment

**Ray Remote Function:** `@ray.remote` decorator on `generate_image_gemini` (app.py:385) enables parallel execution.

**Ray Serve Deployment:** `CampaignGeneratorAgent` class (app.py:683) wraps the app with `@serve.deployment` and `@serve.ingress(app)`.

**Config:** `serve_config.yaml` defines the deployment with 1 replica on port 8000.

## Key Implementation Details

### Image Generation with Gemini
- Model: `gemini-2.5-flash-image-preview`
- Prompt includes aspect ratio hint (e.g., "1080:1920 aspect ratio")
- Returns inline image data extracted from `response.candidates[0].content.parts`
- Images are base64-encoded for HTML embedding

### Structured Output Pattern
Node 3 uses `llm.with_structured_output(GraphicConceptsOutput)` to ensure consistent data structure. This is critical because Node 4 expects a list of `GraphicConcept` objects with specific fields.

### State Management
All nodes return `{**state, "new_field": value, "tracer": tracer}` to preserve state and update specific fields. The `tracer` must always be passed through.

### Error Handling
- Image generation has 3 retries with exponential backoff
- Failed images return error messages in the result dict
- UI shows ❌ for errors, ✅ for successes

## Common Modifications

**Adding a new workflow node:**
1. Define async function with signature `async def my_node(state: CampaignState) -> CampaignState`
2. Update `CampaignState` TypedDict if new fields are needed
3. Add node: `workflow.add_node("my_node", my_node)`
4. Add edge: `workflow.add_edge("previous_node", "my_node")`

**Changing the LLM:**
Update `llm` initialization (app.py:215) and ensure API key is in `.env`

**Adding form fields:**
Add new `F.Input*` to `campaign_form_model` (app.py:75), update `CampaignState`, and access via `inputs.get("field_name")` in runner

**Modifying prompts:**
- Company profile: Edit `company_profile_template` (app.py:11)
- Campaign plan: Edit prompt in Node 2 (app.py:290)
- Graphic concepts: Edit prompt in Node 3 (app.py:333)
