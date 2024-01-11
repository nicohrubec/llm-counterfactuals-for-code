from typing import List


class Parser:
    def parse(self, program: str) -> List[str]:
        raise NotImplementedError

    def unparse(self, split_program: List[str]) -> str:
        raise NotImplementedError
