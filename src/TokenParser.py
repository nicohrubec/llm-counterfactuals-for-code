from typing import List
import re

from Parser import Parser
from helpers import format_code


class TokenParser(Parser):
    """
    Given a program as input, this Parser splits it into a list of tokens.
    """
    def parse(self, program: str) -> List[str]:
        # Split the program by both spaces and newlines
        tokens = re.split(r'[\s]+', program)
        return tokens

    def unparse(self, tokens: List[str]) -> str:
        program = ' '.join(tokens)
        return format_code(program)
