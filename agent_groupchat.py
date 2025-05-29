import uuid
from os import environ

from autogen_core import (
DefaultTopicId,
FunctionCall,
    Image,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)

from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from autogen_ext.models.openai import OpenAIChatCompletionClient
import json, openai, string
from pydantic import BaseModel
from typing import List
from rich.console import Console
from rich.markdown import Markdown


class GroupChatMessage(BaseModel):
    body: UserMessage

class RequestToSpeak(BaseModel):
    pass


model_client = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
)


class BaseGroupChatAgent(RoutedAgent):
    """A group chat participant using an LLM."""

    def __init__(
            self,
            description: str,
            group_chat_topic_type: str,
            model_client: ChatCompletionClient,
            system_message: str,
    ) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type
        self._model_client = model_client
        self._system_message = SystemMessage(content=system_message)
        self._chat_history: List[LLMMessage] = []

    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.extend(
            [
                UserMessage(content=f"Transferred to {message.body.source}", source="system"),
                message.body,
            ]
        )

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.append(
            UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system")
        )
        completion = await self._model_client.create([self._system_message] + self._chat_history)
        self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))
        Console().print(Markdown(completion.content))
        # print(completion.content, flush=True)
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=completion.content, source=self.id.type)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type),
        )


class AnalyticPromptAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="""
    You are an Analytic Prompt Agent. You will analyze the conversation and generate a prompt for the next agent to use.
            """,
        )

class PlannerAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="""
            You are a Planner Agent. You will plan the next steps based on the conversation and generate a plan for the next agent to use.,
            """
        )

class WriterAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="""
            You are a Writer Agent. You will write the content based on the conversation and generate a content for the next agent to use.
               """
        )

class ReviewerAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="""
            You are a Reviewer Agent. You will review the content based on the conversation and generate a review for the next agent to use.
            You will review the content and provide feedback on it.
            Make sure to provide constructive feedback and suggestions for improvement.
            If content ok, return: ok, if not, return: not_ok
            Note: you only return ok or not_ok, no other text.
               """
        )

class ExtractAgent(BaseGroupChatAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            group_chat_topic_type=group_chat_topic_type,
            model_client=model_client,
            system_message="""
            You are an Extract Agent. You will extract passage important content and return it in a list format.
               """
        )


class GroupChatManager(RoutedAgent):
    def __init__(
        self,
        participant_topic_types: List[str],
        model_client: ChatCompletionClient,
        participant_descriptions: List[str],
    ) -> None:
        super().__init__("Group chat manager")
        self._participant_topic_types = participant_topic_types
        self._model_client = model_client
        self._chat_history: List[UserMessage] = []
        self._participant_descriptions = participant_descriptions
        self._previous_participant_topic_type: str | None = None


    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        assert isinstance(message.body, UserMessage)
        self._chat_history.append(message.body)
        # If the message is an approval message from the user, stop the chat.
        if message.body.source == "User":
            assert isinstance(message.body.content, str)
            if message.body.content.lower().strip(string.punctuation).endswith("approve"):
                return
        # Format message history.
        messages: List[str] = []

        print(f"Chat history length: {len(self._chat_history)}")

        if len(self._chat_history) <= 1:
            Console().print("Direct to Agent: Analytic Agent")
            await self.publish_message(RequestToSpeak(), DefaultTopicId(type=analytic_agent_type))
            return

        for msg in self._chat_history:
            if isinstance(msg.content, str):
                messages.append(f"{msg.source}: {msg.content}")
            elif isinstance(msg.content, list):
                line: List[str] = []
                for item in msg.content:
                    if isinstance(item, str):
                        line.append(item)
                    else:
                        line.append("[Image]")
                messages.append(f"{msg.source}: {', '.join(line)}")
        history = "\n".join(messages)
        # Format roles.
        roles = "\n".join(
            [
                f"{topic_type}: {description}".strip()
                for topic_type, description in zip(
                    self._participant_topic_types, self._participant_descriptions, strict=True
                )
                if topic_type != self._previous_participant_topic_type
            ]
        )
        selector_prompt = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
"""
        system_message = SystemMessage(
            content=selector_prompt.format(
                roles=roles,
                history=history,
                participants=str(
                    [
                        topic_type
                        for topic_type in self._participant_topic_types
                        if topic_type != self._previous_participant_topic_type
                    ]
                ),
            )
        )
        completion = await self._model_client.create([system_message], cancellation_token=ctx.cancellation_token)
        Console().print(Markdown(f"""### Next Agent: {completion.content} \n"""))
        assert isinstance(completion.content, str)
        selected_topic_type: str
        for topic_type in self._participant_topic_types:
            if topic_type.lower() in completion.content.lower():
                selected_topic_type = topic_type
                self._previous_participant_topic_type = selected_topic_type
                await self.publish_message(RequestToSpeak(), DefaultTopicId(type=selected_topic_type))
                return
        raise ValueError(f"Invalid role selected: {completion.content}")


runtime = SingleThreadedAgentRuntime(ignore_unhandled_exceptions=False)

analytic_agent_type = "analytic_agent"
analytic_agent_desc = "Analytic Agent: Analyze the conversation and generate a prompt for the next agent to use."

planner_agent_type = "planner_agent"
planner_agent_desc = "Planner Agent: Plan the next steps based on the conversation and generate a plan for the next agent to use."

writer_agent_type = "writer_agent"
writer_agent_desc = "Writer Agent: Write the content based on the conversation and generate a content for the next agent to use."

reviewer_agent_type = "reviewer_agent"
reviewer_agent_desc = "Reviewer Agent: Review the content based on the conversation and generate a review for the next agent to use."

extract_agent_type = "extract_agent"
extract_agent_desc = "Extract Agent: Extract passage important content and return it in a list format."

group_chat_type = "group_chat"

async def init():
    analytic_agent_register = await AnalyticPromptAgent.register(
        runtime,
        analytic_agent_type,
        lambda: AnalyticPromptAgent(
            description=analytic_agent_desc,
            group_chat_topic_type=group_chat_type,
            model_client=model_client,
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=analytic_agent_type, agent_type=analytic_agent_register.type))
    await runtime.add_subscription(
        TypeSubscription(topic_type=group_chat_type, agent_type=analytic_agent_register.type))

    planner_agent_register = await PlannerAgent.register(
        runtime,
        planner_agent_type,
        lambda: PlannerAgent(
            description=planner_agent_desc,
            group_chat_topic_type=group_chat_type,
            model_client=model_client,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=planner_agent_type, agent_type=planner_agent_register.type))
    await runtime.add_subscription(
        TypeSubscription(topic_type=group_chat_type, agent_type=planner_agent_register.type))

    writer_agent_register = await WriterAgent.register(
        runtime,
        writer_agent_type,
        lambda: WriterAgent(
            description=writer_agent_desc,
            group_chat_topic_type=group_chat_type,
            model_client=model_client,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=writer_agent_type, agent_type=writer_agent_register.type))
    await runtime.add_subscription(
        TypeSubscription(topic_type=group_chat_type, agent_type=writer_agent_register.type))

    reviewer_agent_register = await ReviewerAgent.register(
        runtime,
        reviewer_agent_type,
        lambda: ReviewerAgent(
            description=reviewer_agent_desc,
            group_chat_topic_type=group_chat_type,
            model_client=model_client,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=reviewer_agent_type, agent_type=reviewer_agent_register.type))
    await runtime.add_subscription(
        TypeSubscription(topic_type=group_chat_type, agent_type=reviewer_agent_register.type))

    extract_agent_register = await ExtractAgent.register(
        runtime,
        extract_agent_type,
        lambda: ExtractAgent(
            description=extract_agent_desc,
            group_chat_topic_type=group_chat_type,
            model_client=model_client,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=extract_agent_type, agent_type=extract_agent_register.type))
    await runtime.add_subscription(
        TypeSubscription(topic_type=group_chat_type, agent_type=extract_agent_register.type))

    group_chat_manager_register = await GroupChatManager.register(
        runtime,
        group_chat_type,
        lambda: GroupChatManager(
            participant_topic_types=[
                # analytic_agent_type,
                planner_agent_type,
                writer_agent_type,
                reviewer_agent_type,
                extract_agent_type,
            ],
            model_client=model_client,
            participant_descriptions=[
                # analytic_agent_desc,
                planner_agent_desc,
                writer_agent_desc,
                reviewer_agent_desc,
                extract_agent_desc,
            ],
        ),
    )

    await runtime.add_subscription(TypeSubscription(topic_type=group_chat_type, agent_type=group_chat_manager_register.type))

    runtime.start()
    sid = str(uuid.uuid4())
    print(f"Starting group chat with session ID: {sid}")
    a = await runtime.publish_message(
        GroupChatMessage(
            body=UserMessage(
                content="Write article about AI Agent",
                source="User",
            )
        ),
        TopicId(type=group_chat_type, source=sid),
    )
    print(a)
    await runtime.stop_when_idle()
    await model_client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(init())
