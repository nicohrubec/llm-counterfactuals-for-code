from typing import List

from Parser import Parser


class LineParser(Parser):
    """
    Given a program as input this Parser splits it into a list of lines,
    while guaranteeing that there is an empty line in between two lines of code.
    """
    def parse(self, program: str) -> List[str]:
        program = program.split("\n")
        program = "\n\n".join([line for line in program if len(line.strip()) > 0])

        return program.split("\n")

    def unparse(self, split_program: List[str]) -> str:
        return "\n".join(split_program)
