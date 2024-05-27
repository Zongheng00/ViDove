from openai import OpenAI
from .abs_api_model import AbsApiModel

class LLM(AbsApiModel):
    def __init__(self, client, model_name, system_prompt, temp = 0.15) -> None:
        super().__init__()
        self.client = client
        if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
            self.model_name = model_name
        else:
            raise NotImplementedError
        self.system_prompt = system_prompt
        self.temp = temp
    
    def send_request(self, input):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system","content": self.system_prompt},
                {"role": "user", "content": input}
            ],
            temperature=self.temp
        )
        return response.choices[0].message.content.strip()
