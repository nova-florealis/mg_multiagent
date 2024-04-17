from itertools import takewhile

def next_paragraph(it):
    return ''.join(takewhile(lambda x: x.strip(), f)).strip()


def say(s):
    print("START", s, "END")

with open('PrincipiaDiscordia.txt') as f:
     paragraph = next_paragraph(f)
     while paragraph:
         say(paragraph)
         paragraph = next_paragraph(f)
