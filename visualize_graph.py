"""
Visualize the LangGraph workflow for the AI Campaign Generator Agent
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict, Optional
from pydantic import BaseModel, Field

# Import the state definition
class CampaignState(TypedDict):
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
    company_profile: str
    graphic_concepts: Optional[object]
    generated_images: List[Dict]
    tracer: object

# Create the workflow (same as in app.py)
workflow = StateGraph(CampaignState)

# Add nodes
workflow.add_node("company_profile", lambda x: x)
workflow.add_node("graphic_concepts", lambda x: x)
workflow.add_node("generate_images", lambda x: x)

# Add edges
workflow.add_edge(START, "company_profile")
workflow.add_edge("company_profile", "graphic_concepts")
workflow.add_edge("graphic_concepts", "generate_images")
workflow.add_edge("generate_images", END)

# Compile the graph
graph = workflow.compile()

# Generate visualization
try:
    # Get the graph as Mermaid diagram
    print("LangGraph Workflow Visualization")
    print("=" * 50)
    print("\nMermaid Diagram:")
    print("-" * 50)
    print(graph.get_graph().draw_mermaid())
    print("-" * 50)

    # Try to generate PNG if graphviz is available
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("workflow_graph.png", "wb") as f:
            f.write(png_data)
        print("\n✓ PNG visualization saved to: workflow_graph.png")
    except Exception as e:
        print(f"\n✗ Could not generate PNG (install graphviz): {e}")

    # ASCII representation
    print("\nASCII Workflow:")
    print("-" * 50)
    print("""
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────────────────┐
    │  company_profile    │
    │  (Node 1)           │
    │                     │
    │ - Uses Jinja2       │
    │   template          │
    │ - Calls GPT-4o      │
    │ - Fills gaps with   │
    │   assumptions       │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  graphic_concepts   │
    │  (Node 2)           │
    │                     │
    │ - Structured output │
    │ - 10+ graphics      │
    │ - Table format      │
    │ - Platform/res      │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  generate_images    │
    │  (Node 3)           │
    │                     │
    │ - Ray parallel      │
    │ - Gemini 2.0 Flash  │
    │ - Base64 images     │
    │ - Progress tracking │
    └──────────┬──────────┘
               │
               ▼
         ┌─────────┐
         │   END   │
         └─────────┘
    """)
    print("-" * 50)

    # Node details
    print("\nNode Details:")
    print("-" * 50)
    nodes = graph.get_graph().nodes
    for node_id, node_data in nodes.items():
        print(f"• {node_id}: {node_data}")

    print("\nEdge Details:")
    print("-" * 50)
    edges = graph.get_graph().edges
    for edge in edges:
        print(f"• {edge.source} → {edge.target}")

except Exception as e:
    print(f"Error generating visualization: {e}")
