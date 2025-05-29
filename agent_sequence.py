import uuid
from os import environ
environ["OPENAI_API_KEY"] = "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"

from dataclasses import dataclass
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

# Message class for sequential workflow
@dataclass
class Message:
    content: str

# Initialize OpenAI client
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
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
        super().__init__("Planner Agent - Creates execution plans")
        self._system_message = SystemMessage(
            content=(
                "You are a Planner Agent. Your role is to:\n"
                "- Review the analysis from the previous agent\n"
                "- Create a detailed step-by-step plan\n"
                "- Define the structure and approach for content creation\n"
                "- Set clear objectives for the writing phase\n\n"
                "Output a comprehensive plan with numbered steps and clear objectives."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        console.print(Panel(
            f"[bold yellow]PlannerAgent[/bold yellow] is creating execution plan...",
            title="📝 Planning Phase",
            border_style="yellow"
        ))
        
        prompt = f"Based on this analysis, create a detailed plan:\n\n{message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        
        console.print(Panel(
            response,
            title="📋 Execution Plan",
            border_style="green"
        ))

        # Send to next agent: WriterAgent
        await self.publish_message(
            Message(response), 
            topic_id=TopicId(writer_topic_type, source=self.id.key)
        )

# WriterAgent - Third in sequence
@type_subscription(topic_type=writer_topic_type)
class WriterAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Writer Agent - Creates content")
        self._system_message = SystemMessage(
            content=(
                "You are a Writer Agent. Your role is to:\n"
                "- Follow the plan provided by the Planner Agent\n"
                "- Create high-quality, engaging content\n"
                "- Ensure the content meets all specified requirements\n"
                "- Write in a clear, professional, and engaging style\n\n"
                "Output well-structured, comprehensive content based on the plan."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        console.print(Panel(
            f"[bold green]WriterAgent[/bold green] is writing content...",
            title="✍️ Writing Phase",
            border_style="green"
        ))
        
        prompt = f"Write content based on this plan:\n\n{message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        
        console.print(Panel(
            response,
            title="📄 Written Content",
            border_style="green"
        ))

        # Send to next agent: ReviewerAgent
        await self.publish_message(
            Message(response), 
            topic_id=TopicId(reviewer_topic_type, source=self.id.key)
        )

# ReviewerAgent - Fourth in sequence
@type_subscription(topic_type=reviewer_topic_type)
class ReviewerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("Reviewer Agent - Reviews and provides feedback")
        self._system_message = SystemMessage(
            content=(
                "You are a Reviewer Agent. Your role is to:\n"
                "- Carefully review the content created by the Writer Agent\n"
                "- Check for quality, accuracy, and completeness\n"
                "- Provide constructive feedback and suggestions\n"
                "- Evaluate if the content meets the original requirements\n\n"
                "Provide detailed feedback and your reviewed/improved version of the content."
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        console.print(Panel(
            f"[bold red]ReviewerAgent[/bold red] is reviewing content...",
            title="🔍 Review Phase",
            border_style="red"
        ))
        
        prompt = f"Review and improve this content:\n\n{message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        assert isinstance(response, str)
        
        console.print(Panel(
            response,
            title="📝 Review & Feedback",
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

        # Send to UserAgent to complete workflow
        await self.publish_message(
            Message(response), 
            topic_id=TopicId(user_topic_type, source=self.id.key)
        )

# UserAgent - Final recipient to display results
@type_subscription(topic_type=user_topic_type)
class UserAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("User Agent - Displays final results")

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        console.print(Panel(
            f"[bold green]✅ Sequential Workflow Completed![/bold green]\n\n"
            f"Final Result:\n{message.content}",
            title="🎉 Workflow Complete",
            border_style="bright_green"
        ))
        
        console.print("\n" + "="*80)
        console.print("[bold cyan]📊 WORKFLOW SUMMARY:[/bold cyan]")
        console.print("1. ✅ AnalyticPromptAgent - Analyzed the request")
        console.print("2. ✅ PlannerAgent - Created execution plan")
        console.print("3. ✅ WriterAgent - Generated content")
        console.print("4. ✅ ReviewerAgent - Reviewed and improved content")
        console.print("5. ✅ ExtractAgent - Extracted key information")
        console.print("6. ✅ UserAgent - Displayed final results")
        console.print("="*80)

# Main initialization function
async def init():
    runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=False)
    
    console.print(Panel(
        "[bold blue]🚀 Starting Sequential Workflow[/bold blue]\n"
        "Flow: AnalyticPromptAgent → PlannerAgent → WriterAgent → ReviewerAgent → ExtractAgent → UserAgent",
        title="Sequential AI Agent Workflow",
        border_style="bright_blue"
    ))
    
    # Register all agents (type_subscription decorator handles subscriptions automatically)
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
    
    await UserAgent.register(
        runtime, user_topic_type, lambda: UserAgent()
    )
    
    # Start the runtime
    runtime.start()
    
    # Start the workflow by sending initial message to AnalyticPromptAgent
    sid = str(uuid.uuid4())
    console.print(f"\n[bold yellow]🎯 Starting workflow with session ID: {sid}[/bold yellow]")
    console.print("[bold cyan]📝 Initial Request: 'Write article about AI Agent'[/bold cyan]\n")
    
    await runtime.publish_message(
        Message("Write article about AI Agent"),
        TopicId(analytic_topic_type, source=sid)
    )
    
    # Wait for workflow to complete
    await runtime.stop_when_idle()
    await model_client.close()
    
    console.print("\n[bold green]🎊 Sequential Workflow Finished Successfully![/bold green]")

if __name__ == "__main__":
    import asyncio
    asyncio.run(init())


