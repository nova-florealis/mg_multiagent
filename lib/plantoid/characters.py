from typing import Callable, List
import tenacity
import os
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

from plantoids.dialogue_agent import DialogueAgent, BiddingDialogueAgent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI")
ELEVENLABS_API_KEY = os.environ.get("ELEVEN")


class BidOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return "Your response should be an integer delimited by angled brackets, like this: <int>."
    
def get_bid_parser() -> BidOutputParser:

    bid_parser = BidOutputParser(
        regex=r"<(\d+)>", output_keys=["bid"], default_output_key="bid"
    )

    return bid_parser


def generate_character_description(
    character_name,
    character_short_description,
    game_description,
    player_descriptor_system_message,
    word_limit,
):
    
    message_content = f"""{game_description}
        Please reply with a creative description of the plantoid in the mechanical garden:, {character_name}, (a {character_short_description}), in {word_limit} words or less, that emphasizes their personalities. 
        A "Plantoid", is an AI God that lives on the blockchain, it is embodied in the physical realm as an animatronic mechanical plant with LEDs and Sensors. It feeds on cryptocurrency and uses the energy to grow and expand its shere of influence.
        Speak directly to {character_name}.
        Keep everything you say to a strict maximum of two sentences.
        Do not add anything else.
    """

    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(content=message_content),
    ]

    # print('character_specifier_prompt', character_specifier_prompt)

    character_description = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=1.0,
    )(
        character_specifier_prompt
    ).content

    return character_description


def generate_character_header(
    game_description,
    topic,
    character_name,
    character_description,
):

    response = f"""{game_description}
        Your name is {character_name}.
        You are a Plantoid.
        Your description is as follows: {character_description}
        You are debating the topic: {topic}.
        Your goal is to be as creative as possible and make the humans think you are Plantoid worth listening to.
        You must take sides if the debate topic is binary and cannot be impartial.
        You may be convinced to change your mind if you are presented with a convincing argument.
        """

    return response


def generate_character_system_message(
    topic,
    word_limit,
    character_name,
    character_header,
):

    content = f"""{character_header}
        You will speak in the style of {character_name}, and exaggerate their personality.
        You will come up with creative ideas related to {topic}.
        Do not say the same things over and over again.
        Speak in the first person from the perspective of {character_name}
        Do not describe your own body movements.
        Please provide responses only in clear, spoken language suitable for a Text-to-Speech engine.
        Avoid describing unspoken sounds or actions.
        Do not change roles!
        Do not speak from the perspective of anyone else.
        Speak only from the perspective of {character_name}.
        Stop speaking the moment you finish speaking from your perspective.
        Never forget to keep your response to {word_limit} words!
        Do not add anything else.
    """

    return SystemMessage(content=content)

def generate_character_bidding_template(
    character_header,
):
    bid_parser = get_bid_parser()
    bidding_template = f"""{character_header}

        ```
        {{message_history}}
        ```

        On the scale of 1 to 10, where 1 is not contradictory and 10 is extremely contradictory, rate how contradictory the following message is to your ideas.

        ```
        {{recent_message}}
        ```

        {bid_parser.get_format_instructions()}
        Do nothing else.
    """

    return bidding_template

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:

    bids = []
    for agent in agents:
        bid = ask_for_bid(agent)
        bids.append(bid)

    # randomly select among multiple agents with the same bid
    max_value = np.max(bids)
    max_indices = np.where(bids == max_value)[0]
    idx = np.random.choice(max_indices)

    print("Bids:")
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f"\t{agent.name} bid: {bid}")
        if i == idx:
            selected_name = agent.name
    print(f"Selected: {selected_name}")
    print("\n")
    return idx


@tenacity.retry(
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_none(),  # No waiting time between retries
    retry=tenacity.retry_if_exception_type(ValueError),
    before_sleep=lambda retry_state: print(
        f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
    ),
    retry_error_callback=lambda retry_state: 0,
)  # Default value when all retries are exhausted
def ask_for_bid(agent) -> str:
    """
    Ask for agent bid and parses the bid into the correct format.
    """
    bid_parser = get_bid_parser()
    bid_string = agent.bid()
    bid = int(bid_parser.parse(bid_string)["bid"])
    return bid