import os
from typing import Callable, List
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI")
ELEVENLABS_API_KEY = os.environ.get("ELEVEN")


def specify_topic_from_prompt(
    game_description,
    word_limit,
    character_names,
):

    topic_specifier_prompt = [
        SystemMessage(content="You can make a task more specific."),
        HumanMessage(
            content=f"""{game_description}
            
            You are an evil debate moderator.
            Help us elaborate on the debate topic and make it more specific. 
            Do not shy away from controversy. We like to shock and polarize people.
            Keep it simple.
            Please reply with the specified topic in {word_limit} words or less. 
            Speak directly to the participants: {*character_names,}.
            Conclude by asking for an opening statement."""
        ),
    ]
    #TODO: read the prompt

    specified_topic = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=1.0,
    )(topic_specifier_prompt).content

    return specified_topic