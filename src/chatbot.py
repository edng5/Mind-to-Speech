from dotenv import load_dotenv
import os
import anthropic
import yaml

load_dotenv()

class Chatbot:
    """
    Simple wrapper around the Anthropics Claude API.
    Expects ANTHROPIC_API_KEY to be set in the environment (e.g. via .env).
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Client(api_key=api_key)

    def send_message(self, name: str, message: str) -> str:
        """
        Send a single-turn message to Claude and return the assistant reply.
        """
        with open("prompt-template.yml", "r") as file:
            prompt_template = yaml.safe_load(file)  
        system_prompt = prompt_template['prompts']['system']['content'].replace("{{name}}", name)
        user_prompt = prompt_template['prompts']['user']['content'].replace("{{user_input}}", message)
        prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = self.client.messages.create(
            max_tokens=1000,
            model="claude-haiku-4-5",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return resp.content[0].text
