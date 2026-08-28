import os
import time
import dotenv
import openai
import json


class LLM():

    def __init__(self, model):
        self._model  = model
        self._client = None


    def set_up(self):
        dotenv.load_dotenv()
        self._client = openai.OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key  = os.getenv("OPENROUTER_API_KEY"),
        )


    def respond(self, chat, tools=None):
        completion = self._complete_chat(chat, tools)
        tool_calls = completion.choices[0].message.tool_calls
        response   = completion.choices[0].message.content
        response   = self._clean_response_from_special_chars(response)
        return response, tool_calls


    def _complete_chat(self, chat, tools, max_attempts=10):
        backoff = 2 # Initial backoff time in seconds

        for attempt in range(1, max_attempts+1):
            try:
                if tools is not None:
                    completion = self._client.chat.completions.create(
                        model    = self._model,
                        messages = chat,
                        tools    = tools
                    )
                else:
                    completion = self._client.chat.completions.create(
                        model    = self._model,
                        messages = chat
                    )
                if completion.choices[0].message.content    is None and \
                   completion.choices[0].message.tool_calls is None:
                    raise AssertionError("Completion is None")
                return completion

            except AssertionError:
                continue

            except openai.AuthenticationError as e:
                raise RuntimeError("Authentication failed") from e

            except openai.RateLimitError as e:
                time.sleep(min(backoff**attempt,60))
                continue

            except openai.APIStatusError as e:
                status = e.status_code

                if status in {400, 401, 402, 403, 404}:
                    raise RuntimeError(f"API error with status {status}") from e

                if status in {408, 429, 500, 502, 503}:
                    time.sleep(min(backoff**attempt,60))
                    continue

                raise RuntimeError(f"Unexpected API error with status {status}") from e

            except openai.APIError as e:
                time.sleep(min(backoff**attempt,60))
                continue

            except openai.OpenAIError as e:
                time.sleep(min(backoff**attempt,60))
                continue

            except (ValueError, json.JSONDecodeError) as e:
                time.sleep(min(backoff**attempt, 60))
                continue

            except Exception as e:
                raise RuntimeError("Unexpected runtime error") from e

        raise RuntimeError(f"Failed to complete chat after {max_attempts} attempts.")


    def _clean_response_from_special_chars(self, response):
        if response is None:
            return
        
        allowed_non_alnum_chars = [" ", ",", ".", ";", ":", "!", "%", "&", "?", "|", "-", "_", "/",
                                   '"', "'", "´", "`", "{", "}", "[", "]", "(", ")", "<", ">", "=",
                                   "+", "*", "#", "\n", "\\", "~", "@", "^", "$", "\t", "\r"]
        cleaned_response = []
        for char in response:
            if char.isalnum() or char in allowed_non_alnum_chars:
                cleaned_response.append(char)
        return "".join(cleaned_response)
