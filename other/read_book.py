import os
from elevenlabs import generate, stream, set_api_key
from itertools import takewhile
import playsound

ELEVENLABS_API_KEY ="<YOUR_ELEVENLABS_KEY>"
set_api_key(ELEVENLABS_API_KEY)

def next_paragraph(it):
    return ''.join(takewhile(lambda x: x.strip(), f)).strip()

def say(s):
    # text = f"What shall we debate?"

    use_narrator_voice_id = "<YOUR_VOICELAB_ID>"

    audio_stream = generate(
        text=s,
        model="eleven_turbo_v2",
        voice=use_narrator_voice_id,
        stream=True
    )

    stream(audio_stream)
    # playsound(audio_stream)
    # print("START", s, "END")

if __name__ == "__main__":

    print("Read book...")

    with open(os.getcwd()+'/other/PrincipiaDiscordia.txt') as f:
        i = 0
        paragraph = next_paragraph(f)
        while paragraph:
            print("Chunk:", i)
            say(paragraph)
            paragraph = next_paragraph(f)

            i += 1








