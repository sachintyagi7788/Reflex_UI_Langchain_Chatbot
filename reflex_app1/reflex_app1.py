import reflex as rx
import os
import asyncio
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_logic.qa_handler import create_qa_chain, get_answer_from_chain

PDF_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', "your_document.pdf")

_global_qa_chain = None # Global variable to hold the QA chain

class ChatState(rx.State):
    question: str = ""
    chat_history: list[tuple[str, str]] = []
    is_loading: bool = False
    pdf_loaded: bool = False
    error_message: str = ""

    async def load_pdf_and_init_chain(self):
        global _global_qa_chain

        # if not os.getenv("OPENAI_API_KEY"):
        #     self.error_message = "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
        #     print(self.error_message)
        #     return

        if not os.path.exists(PDF_FILE_PATH):
            self.error_message = f"Error: PDF file not found at {PDF_FILE_PATH}"
            print(self.error_message)
            return

        self.is_loading = True
        self.error_message = ""
        self.pdf_loaded = False
        yield

        try:
            print(f"Attempting to create QA chain with PDF: {PDF_FILE_PATH}")
            loop = asyncio.get_event_loop()
            _global_qa_chain = await loop.run_in_executor(None, create_qa_chain, PDF_FILE_PATH)
            self.pdf_loaded = True
            print("QA Chain loaded successfully into backend variable.")
        except FileNotFoundError as fnf_err:
            self.error_message = str(fnf_err)
            print(f"Error during chain creation: {self.error_message}")
            _global_qa_chain = None
        except ValueError as val_err:
            self.error_message = str(val_err)
            print(f"Error during chain creation: {self.error_message}")
            _global_qa_chain = None
        except Exception as e:
            self.error_message = f"An unexpected error occurred: {e}"
            print(f"Error during chain creation: {self.error_message}")
            _global_qa_chain = None
        finally:
            self.is_loading = False

    async def answer_question(self):
        global _global_qa_chain

        if not self.question.strip():
            return
        if _global_qa_chain is None:
            self.error_message = "QA chain not loaded. Please load the PDF first."
            return
        if self.is_loading:
            return

        self.is_loading = True
        user_question = self.question
        self.chat_history.append((user_question, ""))
        self.question = ""
        self.error_message = ""
        yield

        try:
            print(f"Getting answer for: {user_question}")
            answer = await get_answer_from_chain(_global_qa_chain, user_question)
            self.chat_history[-1] = (user_question, answer)
            print("Answer received and chat history updated.")
        except Exception as e:
            self.error_message = f"Error getting answer: {e}"
            self.chat_history[-1] = (user_question, "Error fetching answer.")
            print(f"Error during answer retrieval: {self.error_message}")
        finally:
            self.is_loading = False

def message_bubble(text: str, is_user: bool, key: str):
    return rx.box(
        rx.markdown(text),
        bg=rx.cond(is_user, rx.color("gray", 4), rx.color("blue", 4)),
        padding="1em", # This "1em" is for padding, which can be an 'em' value.
        border_radius="lg",
        align_self=rx.cond(is_user, "flex-end", "flex-start"),
        max_width="70%",
        key=key,
    )

def index():
    return rx.container(
        rx.vstack(
            rx.cond(
                ChatState.error_message,
                rx.callout(ChatState.error_message, icon="alert_triangle", color_scheme="red", role="alert")
            ),
            rx.button(
                "Load PDF and Initialize QA Bot",
                on_click=ChatState.load_pdf_and_init_chain,
                is_loading=ChatState.is_loading,
                is_disabled=ChatState.pdf_loaded,
                color_scheme="green"
            ),
            rx.cond(
                ChatState.pdf_loaded,
                rx.text("PDF Loaded. Ask your questions below!", color="green", margin_top="0.5em"),
            ),
            rx.box( # Chat history box
                rx.vstack(
                    rx.foreach(
                        ChatState.chat_history,
                        lambda message, index: rx.vstack(
                            message_bubble(message[0], is_user=True, key=f"user_{index}"),
                            message_bubble(message[1], is_user=False, key=f"ai_{index}"),
                            spacing="4",  # Changed from "0.5em" to a valid token
                            width="100%",
                            align_items="stretch",
                        )
                    ),
                    spacing="5", # Changed from "1em" to a valid token
                    width="100%",
                    height="60vh",
                    overflow_y="auto",
                    padding="1em", # This is for padding, which can be an 'em' value.
                    border="1px solid #ddd",
                    border_radius="md"
                ),
                width="100%",
                margin_top="1em"
            ),
            rx.hstack(
                rx.input(
                    placeholder="Ask a question...",
                    value=ChatState.question,
                    on_change=ChatState.set_question,
                    # Corrected code
                    on_key_down=lambda event: rx.cond(
                        event == "Enter",  # Compare the event argument directly
                        ChatState.answer_question,
                        None
                    ),
                    flex_grow=1,
                    is_disabled=ChatState.is_loading | ~ChatState.pdf_loaded
                ),
                rx.button(
                    "Ask",
                    on_click=ChatState.answer_question,
                    is_loading=ChatState.is_loading,
                    is_disabled=ChatState.is_loading | ~ChatState.pdf_loaded
                ),
                width="100%",
                padding_top="1em",
            ),
            align="center",
            spacing="5" # Changed from "1em" to a valid token
        ),
        max_width="800px",
        padding="2em"
    )

app = rx.App(
    theme=rx.theme(
        appearance="light", accent_color="blue"
    )
)
app.add_page(index, title="PDF Chatbot")