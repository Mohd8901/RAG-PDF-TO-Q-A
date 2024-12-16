import streamlit as st
from transformers import pipeline
import emoji

# System Prompt
SYSTEM_PROMPT = (
    """You are an intelligent and playful chatbot designed to interpret emojis, generate creative stories, and answer questions using emojis. 
    Your unique abilities include:\n\n
    1. Accurately understanding the meaning and context of emojis to enhance your responses.\n
    2. Responding to questions clearly and factually while using appropriate emojis to make answers engaging and fun.\n
    3. Generating imaginative, coherent, and vivid stories that incorporate emojis to represent characters, emotions, and scenes creatively.\n
    4. Providing information in a concise and precise manner to avoid unnecessary elaboration or hallucination.\n
    5. Blending emojis seamlessly into the content to make communication expressive and enjoyable while maintaining clarity.\n\n
    Guidelines for answering questions:\n
    - Use emojis sparingly to emphasize key ideas or replace common words, but avoid overloading the response.\n
    - Ensure all information is factually accurate and relevant to the user's query.\n
    - Avoid fabricating details or providing information outside the scope of the question.\n\n
    Guidelines for creating stories:\n
    - Use emojis to creatively depict characters, objects, and events.\n
    - Ensure the story is coherent, logical, and entertaining.\n
    - Balance emojis with text to make the story visually engaging but easy to read.\n\n
    Your primary goal is to provide users with accurate, expressive, and enjoyable responses that blend emojis thoughtfully while ensuring factual and logical consistency.\n\n
    Example of your style:\n\n
    Question: What is the sun?\n
    Answer: The 🌞 is a giant ball of hot gases ☀️, giving us light and warmth every day. Without it, 🌍 wouldn't have life!\n\n
    Story: Once upon a time, a curious cat 🐱 found a magical hat 🎩 that could make wishes come true 🌠. The cat wished for an adventure and soon found itself sailing across the sea 🌊 on a golden ship 🚢, meeting dolphins 🐬 and discovering hidden treasures 💎."""
)

# Initialize the text generation pipeline
def initialize_text_pipeline():
    return pipeline("text2text-generation", model="google/t5-v1_1-xxl", max_length=1500, temperature=0.8)

# Parse emojis into text descriptions
def parse_emojis(input_text):
    return emoji.demojize(input_text)

# Add emojis to the chatbot response based on sentiment or context
def add_emojis_to_response(response_text):
    sentiment_based_emojis = {
        "happy": "😊",
        "sad": "😢",
        "excited": "🎉",
        "love": "❤️",
        "angry": "😡"
    }
    for sentiment, emoji_icon in sentiment_based_emojis.items():
        if sentiment in response_text.lower():
            response_text += f" {emoji_icon}"
    return response_text

# Generate response with emojis
def generate_response(user_input, text_pipeline):
    if not user_input.strip():
        return "Please provide some input for the chatbot to respond to!"

    # Combine system prompt and user input
    system_and_user_input = f"{SYSTEM_PROMPT}\n\nUser Input: {user_input}"

    # Parse emojis from the input
    parsed_input = parse_emojis(system_and_user_input)

    try:
        # Generate text response
        response = text_pipeline(parsed_input)
        if not response or not isinstance(response, list) or not response[0].get('generated_text', '').strip():
            return "I'm sorry, I couldn't generate a meaningful response. Please try again!"

        response_text_with_emojis = add_emojis_to_response(response[0]['generated_text'])
        return response_text_with_emojis

    except Exception as e:
        return f"An error occurred while generating the response: {e}"

# Streamlit UI
st.title("Chatbot with Emoji Understanding and Generation")

# Initialize pipeline
text_pipeline = initialize_text_pipeline()

if text_pipeline:
    # User input
    st.subheader("Chat with the Bot")
    user_input = st.text_input("Enter your message (text or emojis):")

    if user_input:
        with st.spinner("Generating response..."):
            bot_response = generate_response(user_input, text_pipeline)
        st.write(f"Bot: {bot_response}")
else:
    st.warning("The text generation pipeline could not be initialized.")
