from kodosumi.core import forms as F

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


