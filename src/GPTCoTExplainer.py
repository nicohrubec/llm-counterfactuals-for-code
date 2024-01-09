from openai import OpenAI

from helpers import build_user_prompt, extract_code_from_string, extract_questions
from Explainer import Explainer


class GPTCoTExplainer(Explainer):
    def __init__(self, model_str):
        self.model = model_str
        self.client = OpenAI()

    def ask_gpt(self, messages):
        messages = [{"role": "system", "content": self.system_prompt}] + messages
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return completion.choices[0].message.content

    def explain(self, sample: str, target: bool) -> str:
        prompt, prompt_len = build_user_prompt(sample, target)

        messages = [
            {"role": "user",
             "content": f"Which are three questions Q1, Q2, and Q3 whose answers would be most helpful to solve the following problem:\n {prompt}\n"
                        f"Only ask questions that can be answered by looking at the given code. In particular you cannot ask about the blackbox-classifier."}
        ]
        answer_food_for_thought_questions = self.ask_gpt(messages)
        q1, q2, q3 = extract_questions(answer_food_for_thought_questions)
        print(answer_food_for_thought_questions)

        # ask first question
        # use instruction generating prompt in context because sometimes the original question is needed as context to answer the food for thought questions
        messages.append({"role": "assistant", "content": answer_food_for_thought_questions})
        messages.append({"role": "user", "content": q1})
        answer_q1 = self.ask_gpt(messages)
        print(answer_q1)

        # ask second question
        # add answer to first question and second question to context
        messages.append({"role": "assistant", "content": answer_q1})
        messages.append({"role": "user", "content": q2})
        answer_q2 = self.ask_gpt(messages)
        print(answer_q2)

        # ask third question
        messages.append({"role": "assistant", "content": answer_q2})
        messages.append({"role": "user", "content": q3})
        answer_q3 = self.ask_gpt(messages)
        print(answer_q3)

        # ask actual question
        messages.append({"role": "assistant", "content": answer_q3})
        messages.append({"role": "user", "content": prompt})
        answer = self.ask_gpt(messages)
        print(answer)

        explanation = extract_code_from_string(answer)

        return explanation
