import gradio as gr
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ollama embedding
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ollama
Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)

def chat(
        user_message,
        history
    ):
    query = history[-1][0]

    storage_context = StorageContext.from_defaults(persist_dir="store")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query(query)
    for text in response.response_gen:
        history[-1][1] += text
        yield history
    
def user(user_message, history):
        return "", history + [[user_message, ""]]

def create_gradio_interface():
    custom_css = """
    .contain { display: flex; flex-direction: column; }

    #component-0 { height: 100%; }

    #main-container { display: flex; height: 100%; justify-content: center;     }

    #col { height: calc(100vh - 100px); max-width: 50vw; }

    #chatbot { flex-grow: 1; overflow: auto; }

    """
    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        with gr.Row(elem_id="main-container"):
            with gr.Column(scale=1, elem_id="col"):
                chatbot = gr.Chatbot(
                    label="Chat History", 
                    elem_id="chatbot",
                    value=[(None,"Feel free to ask me anything about the document")]
                )
                with gr.Row():
                    query = gr.Textbox(
                        label="Input",
                        placeholder="Enter your query here...",
                        elem_id="query-input",
                        scale=3
                    )
                    query_btn = gr.Button("Send Query", variant="primary")


        query.submit(user, [query, chatbot], [query, chatbot], queue=False).then(
            chat, [query, chatbot], [chatbot]
        )
        query_btn.click(user, [query, chatbot], [query, chatbot], queue=False).then(
            chat, [query, chatbot], [chatbot]
        )

    return demo.queue()


demo = create_gradio_interface()
app = demo.app

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)