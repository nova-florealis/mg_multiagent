import os
from typing import Callable, List
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

from dotenv import load_dotenv
from lib.plantoid.llms import get_llm
from util.util import load_config

# Load environment variables from .env file
load_dotenv()

config = load_config(os.getcwd()+"/configuration.toml")


OPENAI_API_KEY = os.environ.get("OPENAI")
ELEVENLABS_API_KEY = os.environ.get("ELEVEN")

# instantiate the LLM to use
use_interface = config['general']['use_llm']
llm = get_llm(interface=use_interface)

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
            Help us make the debate topic more specific. 
            Do not shy away from controversy. We like to shock and polarize people.
            Keep it brief and simple.
            Please reply with the specified topic in {word_limit} words or less. 
            Name the participants in the debate: {*character_names,}.
            Conclude by asking for an opening statement."""
        ),
    ]
    #TODO: read the prompt

    # specified_topic = ChatOpenAI(
    #     openai_api_key=OPENAI_API_KEY,
    #     temperature=1.0,
    # )(topic_specifier_prompt).content

    specified_topic = llm(topic_specifier_prompt).content

    return specified_topic