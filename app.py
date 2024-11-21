import gradio as gr
from dotenv import load_dotenv
import worker

# Load environment variables
load_dotenv()


def process_document(file):
    """
    Process the uploaded PDF document
    """
    if file is None:
        return "Please upload a PDF file."

    try:
        worker.process_document(file.name)
        return f"Document '{file.name}' has been successfully processed!"
    except Exception as e:
        return f"Error processing document: {str(e)}"


def chat_with_pdf(message, history):
    """
    Handle chat interaction with the PDF
    """
    try:
        # Process the user prompt
        response, updated_history = worker.process_prompt(message)
        return updated_history, response
    except Exception as e:
        return history, f"An error occurred: {str(e)}"


def create_gradio_interface():
    """
    Create the Gradio interface for the PDF assistant
    """
    with gr.Blocks() as demo:
        gr.Markdown("# 📚 Personal PDF Assistant")

        with gr.Tab("Document Upload"):
            with gr.Row():
                pdf_upload = gr.File(file_types=[".pdf"], label="Upload PDF")
                process_status = gr.Textbox(label="Processing Status", interactive=False)

            process_btn = gr.Button("Process Document")
            process_btn.click(process_document, inputs=pdf_upload, outputs=process_status)

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat with PDF Assistant", height=500)
            message_input = gr.Textbox(label="Your Message")
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear Chat")

            send_btn.click(chat_with_pdf, inputs=[message_input, chatbot], outputs=[chatbot, message_input])
            clear_btn.click(lambda: [], inputs=None, outputs=chatbot)

    return demo


def main():
    # Initialize the LLM and embeddings
    worker.init_llm()

    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=8000)


if __name__ == "__main__":
    main()
