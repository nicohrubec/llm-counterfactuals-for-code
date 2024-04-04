from typing import List

from Parser import Parser


class LineParser(Parser):
    """
    Given a program as input this Parser splits it into a list of lines.
    """
    def parse(self, program: str) -> List[str]:
        program = program.split("\n")
        return [line for line in program if len(line.strip()) > 0]

    def unparse(self, split_program: List[str]) -> str:
        return "\n".join(split_program)
