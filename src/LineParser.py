from typing import List

from Parser import Parser


class LineParser(Parser):
    def parse(self, program: str) -> List[str]:
        split_program = program.split("\n")
        return [line for line in split_program]

    def unparse(self, split_program: List[str]) -> str:
        return "\n".join(split_program)
