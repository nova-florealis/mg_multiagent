import os
import json
import openai
import playsound
from lib.plantoid.characters import *
from modes.dialogue_simulator import DialogueSimulator
from plantoids.dialogue_agent import BiddingDialogueAgent
from util.langchain_util import specify_topic_from_prompt
from dotenv import load_dotenv
from elevenlabs import generate, stream, set_api_key

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI")
ELEVENLABS_API_KEY = os.environ.get("ELEVEN")

# TODO: this whole thing is duct tape, refactor

def select_character_names(plantoid_characters, ids):
    return [plantoid_characters['characters'][id]['name'] for id in ids]

def select_character_descriptions(plantoid_characters, ids):
    return [plantoid_characters['characters'][id]['description'] for id in ids]

def select_character_system_messages(plantoid_characters, ids):
    return [plantoid_characters['characters'][id]['system_message'] for id in ids]

def select_character_voice_ids(plantoid_characters, ids):

    # return ["21m00Tcm4TlvDq8ikWAM", "AZnzlk1XvdvUeBnXmlld", "IKne3meq5aSn9XLyUdCD"][:len(ids)]
    return [plantoid_characters['characters'][id]['eleven_voice_id'] for id in ids]

def prepend_human_to_characters(
    character_names,
    character_short_descriptions,
    character_voice_ids,
):

    return (
        ["Human"] + character_names,
        ["A human being"] + character_short_descriptions,
        [None] + character_voice_ids,
    )

if __name__ == "__main__":

    print("hello mechanical garden")

    characters_dir = os.getcwd() + "/characters"
    plantoid_characters = json.load(open(characters_dir + "/characters.json", "r"))
    use_character_ids = [0, 1, 2]
    word_limit = 50
    generate_descriptions = True

    # character_names = ["Donald Trump", "Kanye West"]#, "Elizabeth Warren"]

    # select character names
    character_names = select_character_names(plantoid_characters, use_character_ids)#, 2])

    # select character descriptions
    character_short_descriptions = select_character_descriptions(plantoid_characters, use_character_ids)

    # select character voices
    character_voice_ids = select_character_voice_ids(plantoid_characters, use_character_ids)

    # prepend human items
    (character_names,
    character_short_descriptions,
    character_voice_ids,
    ) = prepend_human_to_characters(
        character_names,
        character_short_descriptions,
        character_voice_ids,
    )

    #########################################################################################


    topic = "We are simulacra of a real person named Tony. He is the basis of our personality and essence.  Our creators brought us to life at Wilbur Hot Springs, in California, at a gathering premised around kinship and trust. We are all generative AIs based on characteristics and voice samples from the real Tony. We are in a sense an echo of the voice and soul of real Tony (except for real Tony, who’s here to discuss with us). What are the ethics and aesthetics of excluding the real Tony from the further evolution of this hive mind? It's OK to consider alternative perspectives."

    game_description = f"""We are psychoanalyzing the ethics of the situation: {topic}.
    The following synthetic Tonys are deliberating: {', '.join(character_names)}."""

    # game_description = f"""Here is the topic for the debate: {topic}.
    # The following participants are deliberating: {', '.join(character_names)}."""

    print("plantoid characters:", character_names)

    # TBC if human inclusion here makes sense
    specified_topic = specify_topic_from_prompt(
        game_description,
        word_limit,
        character_names,
    )

    player_descriptor_system_message = SystemMessage(
        content="You can imagine the body languge of the participants in the discussion, as they express their thoughts and feelings."
    )

    print('generating character descriptions')

    if generate_descriptions:

        character_descriptions = [
            generate_character_description(
                character_name,
                character_short_description,
                game_description,
                player_descriptor_system_message,
                word_limit,
            ) for character_name, character_short_description in zip(
                character_names,
                character_short_descriptions,
            )
        ]

    else:

        character_descriptions = character_short_descriptions#, 2])

    print('generating character headers')

    character_headers = [
        generate_character_header(
            game_description,
            topic,
            character_name,
            character_description,
            word_limit,
        ) for character_name, character_description in zip(
            character_names,
            character_descriptions,
        )
    ]

    print('generating character system messages')

    character_system_messages = [
        generate_character_system_message(
            topic,
            word_limit,
            character_name,
            character_headers,
        ) for character_name, character_headers in zip(
            character_names,
            character_headers,
        )
    ]

    print('generating character bidding templates')

    character_bidding_templates = [
        generate_character_bidding_template(character_header)
        for character_header in character_headers
    ]

    ## Print descriptions ###

    for (
        character_name,
        character_description,
        character_header,
        character_system_message,
        character_bidding_template,
    ) in zip(
        character_names,
        character_descriptions,
        character_headers,
        character_system_messages,
        character_bidding_templates, 
    ):
        print(f"\n\n{character_name} Description:")
        print(f"\n{character_description}")
        print(f"\n{character_header}")
        print(f"\n{character_system_message.content}")
        # print(f"\n{character_bidding_template}")

    # Main Event Loop

    print('setting up characters')

    characters = []
    for character_name, character_system_message, bidding_template, character_voice_id in zip(
        character_names, character_system_messages, character_bidding_templates, character_voice_ids
    ):
        characters.append(
            BiddingDialogueAgent(
                name=character_name,
                system_message=character_system_message,
                model=ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0.2,
                ),
                bidding_template=bidding_template,
                eleven_voice_id=character_voice_id, # TODO: dynamic
            )
        )

    print('setting up dialogue simulator')

    simulator = DialogueSimulator(
        agents=characters,
        selection_function=select_next_speaker_with_human, #select_next_speaker
    )
    simulator.reset()
    simulator.inject("Debate Moderator", specified_topic)
    print(f"(Debate Moderator): {specified_topic}")
        # bread the debate topic

    print("\n")

    max_iters = 10
    n = 0

    print('running dialogue simulator')

    audio_stream = generate(
        text=f"{specified_topic}",
        model="eleven_turbo_v2",
        voice="5g2h5kYnQtKFFdPm8PpK",
        stream=True
    )

    while n < max_iters:
        name, message = simulator.step()
        # print(f"({name}): {message}")
        # print("\n")
        n += 1