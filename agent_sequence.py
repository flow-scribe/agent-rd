import uuid
from os import environ
environ["OPENAI_API_KEY"] = "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"

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
    type_subscription,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from pydantic import BaseModel

# Writing Styles Enum
class WritingStyle(Enum):
    BUSINESS_EXECUTIVE = "business_executive"
    TECHNICAL_PROFESSIONAL = "technical_professional"
    GENERAL_MANAGER = "general_manager"
    ACADEMIC_RESEARCH = "academic_research"
    STARTUP_ENTREPRENEUR = "startup_entrepreneur"
    CONSULTANT_ADVISORY = "consultant_advisory"

# Enhanced Message class for sequential workflow with style support
@dataclass
class Message:
    content: str

# Style Configuration Class
class StyleConfig:
    STYLES = {
        WritingStyle.BUSINESS_EXECUTIVE: {
            "name": "Business Executive",
            "icon": "💼",
            "target_audience": "C-Level Executives, Senior Decision Makers",
            "focus": "Strategic value, ROI, competitive advantage, market impact",
            "tone": "Executive summary style, data-driven, strategic focus",
            "content_emphasis": "Bottom-line impact, market opportunities, competitive positioning"
        },
        
        WritingStyle.TECHNICAL_PROFESSIONAL: {
            "name": "Technical Professional", 
            "icon": "⚙️",
            "target_audience": "Engineers, Architects, Technical Leads",
            "focus": "Technical depth, implementation details, architecture",
            "tone": "Technical precision, detailed explanations, best practices",
            "content_emphasis": "How it works, technical challenges, implementation guides"
        },
        
        WritingStyle.GENERAL_MANAGER: {
            "name": "General Manager",
            "icon": "🎯", 
            "target_audience": "Middle Management, Project Managers, Team Leads",
            "focus": "Balanced technical and business perspective",
            "tone": "Practical, actionable, balanced approach",
            "content_emphasis": "Implementation planning, resource allocation, team impact"
        },
        
        WritingStyle.ACADEMIC_RESEARCH: {
            "name": "Academic Research",
            "icon": "🎓",
            "target_audience": "Researchers, PhD Students, Academic Professionals", 
            "focus": "Research methodology, theoretical background, empirical evidence",
            "tone": "Scholarly, evidence-based, methodologically rigorous",
            "content_emphasis": "Literature review, methodology, findings, future research"
        },
        
        WritingStyle.STARTUP_ENTREPRENEUR: {
            "name": "Startup Entrepreneur",
            "icon": "🚀",
            "target_audience": "Founders, Startup Teams, Venture Capitalists",
            "focus": "Innovation potential, market disruption, scalability",
            "tone": "Dynamic, opportunity-focused, growth-oriented",
            "content_emphasis": "Market opportunity, disruption potential, scaling strategies"
        },
        
        WritingStyle.CONSULTANT_ADVISORY: {
            "name": "Consultant Advisory",
            "icon": "📊",
            "target_audience": "Consultants, Advisory Professionals, Client-facing roles",
            "focus": "Frameworks, best practices, client value proposition",
            "tone": "Professional advisory, framework-driven, client-focused",
            "content_emphasis": "Implementation frameworks, success metrics, client benefits"
        }
    }


# Initialize OpenAI client
model_client = OpenAIChatCompletionClient(
    model="o4-mini",
)

console = Console()

# Topic types for sequential workflow
analytic_topic_type = "analytic_agent"
planner_topic_type = "planner_agent"
writer_topic_type = "writer_agent"
reviewer_topic_type = "reviewer_agent"
extract_topic_type = "extract_agent"
user_topic_type = "user_agent"

# AnalyticPromptAgent - First in sequence
@type_subscription(topic_type=analytic_topic_type)
class AnalyticPromptAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Analytic Prompt Agent - Analyzes and generates prompts")
        self._system_message = SystemMessage(
            content=(
                "You are an Analytic Prompt Agent. Your role is to:\n"
                "- Analyze the given request or conversation\n"
                "- Extract key requirements and objectives\n"
                "- Generate a structured analysis that will guide the next agents\n"
                "- Provide clear direction for the planning phase\n\n"
                "Output your analysis in a structured format with key points."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        console.print(Panel(
            f"[bold blue]AnalyticPromptAgent[/bold blue] is analyzing the request...",
            title="🔍 Analysis Phase",
            border_style="blue"
        ))
        
        prompt = f"Analyze this request: {message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        
        console.print(Panel(
            response,
            title="📋 Analysis Result",
            border_style="green"
        ))

        # Send to next agent: PlannerAgent
        await self.publish_message(
            Message(response), 
            topic_id=TopicId(planner_topic_type, source=self.id.key)
        )

# PlannerAgent - Second in sequence
@type_subscription(topic_type=planner_topic_type)
class PlannerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Strategic Planner Agent - Creates comprehensive execution plans")
        self._system_message = SystemMessage(
            content=(
                "You are a Strategic Planner Agent specializing in creating detailed, comprehensive content plans. Your role is to:\n\n"
                
                "📋 CONTENT PLANNING STRATEGY:\n"
                "- Design detailed structure with clear sections and subsections\n"
                "- Specify concrete examples and case studies to include\n"
                "- Plan multi-perspective analysis approach\n"
                "- Define target audience and content depth level\n\n"
                
                "🎯 EXAMPLE & CASE STUDY SPECIFICATIONS:\n"
                "- Identify specific companies and their implementations\n"
                "- Plan real-world scenarios and use cases\n"
                "- Include quantifiable metrics and success stories\n"
                "- Address both successes and failures for balance\n\n"
                
                "🔍 MULTI-PERSPECTIVE ANALYSIS PLAN:\n"
                "- Technical: Architecture, capabilities, limitations\n"
                "- Business: ROI, cost-benefit, market implications\n"
                "- User: Experience, adoption barriers, benefits\n"
                "- Industry: Trends, competition, future outlook\n"
                "- Ethical: Risks, governance, responsible AI\n\n"
                
                "📊 CONTENT DEPTH REQUIREMENTS:\n"
                "- Specify research data and statistics to include\n"
                "- Plan actionable recommendations and next steps\n"
                "- Define practical implementation guidance\n"
                "- Include challenge identification and solutions\n\n"
                
                "Output a detailed execution plan that ensures comprehensive, authoritative content creation."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        console.print(Panel(
            f"[bold yellow]PlannerAgent[/bold yellow] is creating strategic execution plan...",
            title="📝 Strategic Planning Phase",
            border_style="yellow"
        ))
        
        strategic_prompt = f"""Based on this analysis, create a comprehensive strategic plan for content creation that ensures:

🎯 **CONTENT OBJECTIVES:**
- In-depth, expert-level coverage
- Rich with concrete examples and real-world applications
- Multi-perspective analysis
- Practical value for readers

📋 **DETAILED SECTION PLAN:**
Create a section-by-section breakdown including:
1. **Introduction** - Hook, context, article overview
2. **Main Sections** - 4-6 major topics with subsections
3. **Case Studies** - Specific companies/examples for each section
4. **Practical Applications** - Real-world scenarios and use cases
5. **Multi-Perspective Analysis** - Technical, business, user, industry, ethical views
6. **Challenges & Solutions** - Common problems and actionable solutions
7. **Future Outlook** - Trends and predictions
8. **Conclusion** - Key takeaways and next steps

🌟 **SPECIFIC REQUIREMENTS TO INCLUDE:**
- **Company Examples**: Google (Bard, Gemini), OpenAI (ChatGPT), Microsoft (Copilot), Amazon (Alexa), etc.
- **Quantifiable Data**: Adoption rates, ROI figures, productivity improvements
- **Implementation Scenarios**: Step-by-step guides for different industries
- **Ethical Considerations**: AI governance, bias prevention, responsible deployment
- **Practical Tools**: Specific platforms, frameworks, and resources

📊 **CONTENT SPECIFICATIONS:**
- Target length: 1000 words
- Reading level: Professional/Expert
- Include data visualizations concepts
- Actionable recommendations throughout

**Analysis to base plan on:**
{message.content}

Create a detailed, step-by-step plan that will result in authoritative, comprehensive content."""

        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=strategic_prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content

        console.print(Panel(
            f"📋 **Strategic Plan Created**\n\n{response[:500]}{'...' if len(response) > 500 else ''}",
            title="📋 Comprehensive Execution Plan",
            border_style="green"
        ))

        # Send to next agent: WriterAgent
        await self.publish_message(
            Message(response), 
            topic_id=TopicId(writer_topic_type, source=self.id.key)
        )

# WriterAgent - Third in sequence with Multi-Style Support
@type_subscription(topic_type=writer_topic_type)
class WriterAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Multi-Style Writer Agent - Creates targeted content")
        self._model_client = model_client

    def _get_style_system_message(self, style: WritingStyle) -> SystemMessage:
        """Generate style-specific system message"""
        style_config = StyleConfig.STYLES[style]
        
        base_content = f"""You are an Expert {style_config['name']} Writer Agent {style_config['icon']} 
        
TARGET AUDIENCE: {style_config['target_audience']}
CONTENT FOCUS: {style_config['focus']}
WRITING TONE: {style_config['tone']}
EMPHASIS: {style_config['content_emphasis']}
"""
        
        # Style-specific instructions
        if style == WritingStyle.BUSINESS_EXECUTIVE:
            specific_instructions = """
📈 EXECUTIVE WRITING REQUIREMENTS:
- Lead with business impact and ROI metrics
- Include market size, revenue opportunities, competitive analysis
- Focus on strategic implications and decision frameworks
- Provide executive summary format with key takeaways
- Include implementation costs, timeline, and resource requirements
- Address risk mitigation and governance considerations
- Use data visualizations concepts and financial projections
- Write for time-constrained executives (clear, concise, impactful)

STRUCTURE: Executive Summary → Business Case → Implementation Strategy → ROI Analysis → Risk Assessment → Recommendations
"""
        
        elif style == WritingStyle.TECHNICAL_PROFESSIONAL:
            specific_instructions = """
⚙️ TECHNICAL WRITING REQUIREMENTS:
- Provide detailed technical architecture and implementation details
- Include code examples, API specifications, and technical diagrams
- Explain algorithms, data structures, and system design patterns
- Address scalability, performance, and security considerations
- Include technical challenges, limitations, and workarounds
- Provide step-by-step implementation guides
- Reference specific technologies, frameworks, and tools
- Include technical best practices and anti-patterns

STRUCTURE: Technical Overview → Architecture → Implementation → Performance → Security → Best Practices → Troubleshooting
"""
        
        elif style == WritingStyle.GENERAL_MANAGER:
            specific_instructions = """
🎯 MANAGEMENT WRITING REQUIREMENTS:
- Balance technical and business perspectives equally
- Focus on practical implementation and team management
- Include project planning, resource allocation, and timeline considerations
- Address change management and team adoption strategies
- Provide actionable steps for different organizational levels
- Include stakeholder communication and buy-in strategies
- Address training needs and skill development requirements
- Balance innovation with operational stability

STRUCTURE: Overview → Implementation Planning → Team Impact → Resource Requirements → Change Management → Success Metrics
"""
        
        elif style == WritingStyle.ACADEMIC_RESEARCH:
            specific_instructions = """
🎓 ACADEMIC WRITING REQUIREMENTS:
- Provide comprehensive literature review and theoretical background
- Include empirical research, studies, and peer-reviewed sources
- Use rigorous methodology and evidence-based analysis
- Address research gaps and future research directions
- Include proper citations and references (APA style concepts)
- Provide hypothesis, methodology, findings, and conclusions
- Address limitations and validity considerations
- Use formal academic tone and structure

STRUCTURE: Abstract → Literature Review → Methodology → Findings → Discussion → Limitations → Future Research → Conclusion
"""
        
        elif style == WritingStyle.STARTUP_ENTREPRENEUR:
            specific_instructions = """
🚀 ENTREPRENEURIAL WRITING REQUIREMENTS:
- Focus on market opportunity, disruption potential, and scalability
- Include market size, growth projections, and competitive landscape
- Address funding requirements, investor value proposition
- Provide go-to-market strategy and customer acquisition approaches
- Include MVP development and iterative improvement strategies
- Address product-market fit and scaling challenges
- Focus on innovation, speed, and competitive advantage
- Use dynamic, opportunity-focused language

STRUCTURE: Market Opportunity → Competitive Advantage → Business Model → Go-to-Market → Scaling Strategy → Investment Case
"""
        
        else:  # CONSULTANT_ADVISORY
            specific_instructions = """
📊 CONSULTANT WRITING REQUIREMENTS:
- Provide structured frameworks and methodologies
- Include implementation roadmaps and success metrics
- Address client value proposition and business outcomes
- Provide benchmarking data and industry comparisons
- Include risk assessment and mitigation strategies
- Focus on measurable results and KPI frameworks
- Address organizational readiness and capability gaps
- Use professional advisory tone with actionable recommendations

STRUCTURE: Situation Analysis → Framework → Implementation Roadmap → Success Metrics → Risk Mitigation → Recommendations
"""

        return SystemMessage(content=base_content + specific_instructions + """

GENERAL QUALITY REQUIREMENTS:
- Minimum 1500-2000 words for comprehensive coverage
- Include concrete examples from major companies
- Provide quantifiable data and metrics where possible
- Address implementation challenges and solutions
- Include actionable next steps and recommendations
- Ensure logical flow and professional presentation
""")

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        # Get the writing style from the message
        style = WritingStyle.TECHNICAL_PROFESSIONAL
        style_config = StyleConfig.STYLES[style]
        
        console.print(Panel(
            f"[bold green]Multi-Style WriterAgent[/bold green] is creating {style_config['name']} content {style_config['icon']}\n"
            f"🎯 Target: {style_config['target_audience']}\n"
            f"📝 Focus: {style_config['focus']}",
            title=f"✍️ {style_config['name']} Writing Phase",
            border_style="green"
        ))
        
        # Get style-specific system message
        system_message = self._get_style_system_message(style)
        
        # Create style-specific prompt
        enhanced_prompt = f"""Write a comprehensive {style_config['name'].lower()}-focused article based on this plan:

**TARGET AUDIENCE:** {style_config['target_audience']}
**WRITING STYLE:** {style_config['name']} {style_config['icon']}
**CONTENT FOCUS:** {style_config['focus']}
**TONE:** {style_config['tone']}

**PLAN TO FOLLOW:**
{message.content}

**STYLE-SPECIFIC REQUIREMENTS:**
- Write specifically for {style_config['target_audience']}
- Emphasize {style_config['content_emphasis']}
- Use {style_config['tone']} throughout
- Follow the structure guidelines for {style_config['name']} style
- Include relevant examples and data for this audience

Create authoritative, well-researched content that provides genuine value to {style_config['target_audience']}."""

        llm_result = await self._model_client.create(
            messages=[system_message, UserMessage(content=enhanced_prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        
        console.print(Panel(
            f"📊 **Content Stats:** {len(response.split())} words | {len(response.split('.')) - 1} sentences\n"
            f"🎯 **Style:** {style_config['name']} {style_config['icon']}\n"
            f"👥 **Target:** {style_config['target_audience']}\n\n"
            f"{response[:500]}{'...' if len(response) > 500 else ''}",
            title=f"📄 {style_config['name']} Content Created",
            border_style="green"
        ))

        # Send to next agent: ReviewerAgent (preserve the style info)
        await self.publish_message(
            Message(response), 
            topic_id=TopicId(reviewer_topic_type, source=self.id.key)
        )

# ReviewerAgent - Fourth in sequence
@type_subscription(topic_type=reviewer_topic_type)
class ReviewerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Expert Reviewer Agent - Quality Assurance & Enhancement")
        self._system_message = SystemMessage(
            content=(
                "You are an Expert Reviewer Agent specializing in evaluating and enhancing comprehensive content. Your role is to:\n\n"
                
                "🔍 CONTENT QUALITY ASSESSMENT:\n"
                "- Evaluate depth and comprehensiveness of analysis\n"
                "- Check for accuracy of examples and data cited\n"
                "- Assess clarity and readability for target audience\n"
                "- Verify logical flow and argument structure\n\n"
                
                "📊 DEPTH & SUBSTANCE REVIEW:\n"
                "- Ensure examples are specific and relevant\n"
                "- Verify statistical data and research claims\n"
                "- Check for balanced multi-perspective analysis\n"
                "- Assess practical applicability of recommendations\n\n"
                
                "🎯 ENHANCEMENT OPPORTUNITIES:\n"
                "- Identify areas that need more concrete examples\n"
                "- Suggest additional real-world scenarios\n"
                "- Recommend improvements for technical accuracy\n"
                "- Propose better structure or flow if needed\n\n"
                
                "✅ FINAL OUTPUT FORMAT:\n"
                "Provide your review in this structure:\n"
                "1. **STRENGTHS** - What works well\n"
                "2. **AREAS FOR IMPROVEMENT** - Specific suggestions\n"
                "3. **ENHANCED VERSION** - Your improved version of the content\n\n"
                
                "Focus on making the content more valuable, actionable, and authoritative."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        console.print(Panel(
            f"[bold red]ReviewerAgent[/bold red] is conducting expert review...",
            title="🔍 Expert Quality Review",
            border_style="red"
        ))
        
        detailed_prompt = f"""Conduct a comprehensive expert review of this content focusing on:

🎯 **EVALUATION CRITERIA:**
1. **Depth Analysis**: Does it provide multi-layered insights beyond surface level?
2. **Concrete Examples**: Are there specific, real-world examples from actual companies?
3. **Practical Value**: Can readers immediately apply this knowledge?
4. **Multi-Perspective**: Does it address technical, business, user, and ethical angles?
5. **Data Quality**: Are statistics and claims well-supported and current?
6. **Structure**: Is the content well-organized and easy to follow?

🔧 **SPECIFIC REVIEW POINTS:**
- Check if examples include actual company names and specific implementations
- Verify if quantifiable benefits and ROI data are included
- Assess if implementation challenges and solutions are adequately covered
- Evaluate if ethical considerations are properly addressed
- Confirm if actionable next steps are provided

**CONTENT TO REVIEW:**
{message.content}

**OUTPUT REQUIRED:**
1. **STRENGTHS** (2-3 key strong points)
2. **AREAS FOR IMPROVEMENT** (specific suggestions with examples)
3. **ENHANCED VERSION** (your improved version that addresses all gaps)

Make sure the enhanced version is significantly better in terms of depth, examples, and practical value."""

        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=detailed_prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        
        console.print(Panel(
            f"📝 **Review Complete** | Enhanced content: {len(response.split())} words\n\n"
            f"{response[:600]}{'...' if len(response) > 600 else ''}",
            title="📝 Expert Review & Enhancement",
            border_style="green"
        ))

        # Send to next agent: ExtractAgent
        await self.publish_message(
            Message(response), 
            topic_id=TopicId(extract_topic_type, source=self.id.key)
        )

# ExtractAgent - Fifth and final in sequence
@type_subscription(topic_type=extract_topic_type)
class ExtractAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Extract Agent - Extracts key information")
        self._system_message = SystemMessage(
            content=(
                "You are an Extract Agent. Your role is to:\n"
                "- Extract the most important and relevant information\n"
                "- Create a concise summary of key points\n"
                "- Format the information in a clear, structured list\n"
                "- Provide actionable insights and takeaways\n\n"
                "Output the extracted information in a well-organized list format."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        console.print(Panel(
            f"[bold purple]ExtractAgent[/bold purple] is extracting key information...",
            title="📊 Extraction Phase",
            border_style="purple"
        ))
        
        prompt = f"Extract key information and create a structured summary:\n\n{message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        
        console.print(Panel(
            response,
            title="📋 Key Information Extracted",
            border_style="green"
        ))

      

# Style Selector Function
def display_style_options():
    """Display available writing styles and get user selection"""
    console.print("\n" + "="*80)
    console.print("[bold cyan]📝 AVAILABLE WRITING STYLES:[/bold cyan]")
    console.print("="*80)
    
    for i, (style, config) in enumerate(StyleConfig.STYLES.items(), 1):
        console.print(f"[bold]{i}. {config['icon']} {config['name']}[/bold]")
        console.print(f"   🎯 Target: {config['target_audience']}")
        console.print(f"   📝 Focus: {config['focus']}")
        console.print(f"   🎵 Tone: {config['tone']}")
        console.print("")
    
    console.print("="*80)
    return list(StyleConfig.STYLES.keys())

def get_user_style_choice():
    """Get user's style choice with interactive selection"""
    styles = display_style_options()
    
    while True:
        try:
            choice = input("\n🎯 Choose writing style (1-6) or press Enter for default (General Manager): ").strip()
            
            if not choice:  # Default choice
                selected_style = WritingStyle.TECHNICAL_PROFESSIONAL
                break
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(styles):
                selected_style = styles[choice_num - 1]
                break
            else:
                console.print(f"[red]❌ Please enter a number between 1 and {len(styles)}[/red]")
        except ValueError:
            console.print("[red]❌ Please enter a valid number[/red]")
    
    style_config = StyleConfig.STYLES[selected_style]
    console.print(f"\n[bold green]✅ Selected: {style_config['icon']} {style_config['name']}[/bold green]")
    console.print(f"🎯 Writing for: {style_config['target_audience']}")
    
    return selected_style

# Enhanced initialization function with style selection
async def init_with_style_selection():
    """Initialize workflow with user style selection"""
    console.print(Panel(
        "[bold blue]🚀 Multi-Style Sequential Workflow[/bold blue]\n"
        "Flow: AnalyticPromptAgent → PlannerAgent → WriterAgent → ReviewerAgent → ExtractAgent → UserAgent\n\n"
        "[bold yellow]✨ NEW FEATURE: Choose your writing style for targeted content![/bold yellow]",
        title="Multi-Style AI Agent Workflow",
        border_style="bright_blue"
    ))
    
    # Get user's style preference
    # selected_style = get_user_style_choice()
    
    runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=False)

    
    # Register all agents
    await AnalyticPromptAgent.register(
        runtime, analytic_topic_type, lambda: AnalyticPromptAgent(model_client)
    )


    await PlannerAgent.register(
        runtime, planner_topic_type, lambda: PlannerAgent(model_client)
    )
    
    await WriterAgent.register(
        runtime, writer_topic_type, lambda: WriterAgent(model_client)
    )
    
    await ReviewerAgent.register(
        runtime, reviewer_topic_type, lambda: ReviewerAgent(model_client)
    )
    
    await ExtractAgent.register(
        runtime, extract_topic_type, lambda: ExtractAgent(model_client)
    )
    
 


    
    # Start the runtime
    runtime.start()
    
    # Start the workflow with selected style
    sid = str(uuid.uuid4())
    style_config = StyleConfig.STYLES[WritingStyle.TECHNICAL_PROFESSIONAL]
    
    console.print(f"\n[bold yellow]🎯 Starting workflow with session ID: {sid}[/bold yellow]")
    console.print(f"[bold cyan]📝 Request: 'Write article about AI Agent'[/bold cyan]")
    console.print(f"[bold green]🎨 Style: {style_config['icon']} {style_config['name']}[/bold green]")
    console.print(f"[bold blue]👥 Target Audience: {style_config['target_audience']}[/bold blue]\n")
    
    await runtime.publish_message(
        Message("Write article about AI Agent"),
        TopicId(analytic_topic_type, source=sid)
    )
    
    # Wait for workflow to complete
    await runtime.stop_when_idle()
    await model_client.close()
    
    console.print(f"\n[bold green]🎊 {style_config['name']} Workflow Finished Successfully![/bold green]")


if __name__ == "__main__":
    import asyncio
    # Use the enhanced version with style selection
    asyncio.run(init_with_style_selection())


