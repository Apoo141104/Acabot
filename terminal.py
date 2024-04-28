from dotenv import load_dotenv
import os
import google.generativeai as genai
from PIL import Image
import spacy
from data import problem_solving_prompts

# Initialize Spacy
nlp = spacy.load("en_core_web_lg")

def get_user_input():
    user_input = input("You: ")
    return user_input

def get_image():
    insert_image = input("Do you want to insert an image? (yes/no): ")
    if insert_image.lower() == "yes":
        image_path = input("Enter image path (e.g., /path/to/image.jpg): ")
        try:
            image = Image.open(image_path)
            return image
        except FileNotFoundError:
            print("Error: Image file not found.")
            return None
    else:
        return None

def get_vision_response(model_vision, image):
    response = model_vision.generate_content(image)
    return response.text

def get_text_response(model_text, user_input, relevant_prompts=None):
    if relevant_prompts:
        user_input += " " + " ".join(relevant_prompts)
    response = model_text.start_chat(history=[]).send_message(user_input, stream=True)
    text_response = ""
    for chunk in response:
        text_response += chunk.text
    return text_response

def extract_meaning(user_input):
    doc = nlp(user_input)
    main_verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
    named_entities = [ent.text for ent in doc.ents]
    return main_verbs, nouns, named_entities

def find_similar_words(word):
    similar_words = []
    word_vector = nlp.vocab[word].vector
    for vocab_word in nlp.vocab:
        if vocab_word.has_vector:
            similarity = word_vector.dot(vocab_word.vector)
            if similarity > 0.7:  # Adjust similarity threshold as needed
                similar_words.append(vocab_word.text)
    return similar_words

def identify_similar_words(user_input):
    similar_words = []
    main_verbs, nouns, named_entities = extract_meaning(user_input)
    for word in main_verbs + nouns + named_entities:
        similar_words.extend(find_similar_words(word))
    return similar_words

def main():
    # Load environment variables (assuming API key is set)
    load_dotenv()

    # Configure Gemini Pro Vision model (assuming API key is set)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model_vision = genai.GenerativeModel("gemini-pro-vision")

    # Configure Gemini Pro model (assuming API key is set)
    model_text = genai.GenerativeModel("gemini-pro")

    # Main interaction loop
    chat_history = []
    while True:
        print("Text Bot: What mathematical problem would you like to work on today?")
        user_input = get_user_input()
        image = get_image()

        if image is not None:
            vision_response = get_vision_response(model_vision, image)
            print("Vision Bot:", vision_response)
            chat_history.append(("Vision Bot", vision_response))
        else:
            similar_words = identify_similar_words(user_input)
            # Determine the appropriate prompts and hints based on user's input
            relevant_prompts = []
            for category, subcategory in problem_solving_prompts.items():
                for key, value in subcategory.items():
                    if key in similar_words:
                        relevant_prompts.extend(value["prompt"])
                        relevant_prompts.append(value["hints"])
        
            # Provide prompts and hints if any are relevant
            if relevant_prompts:
                print("Text Bot: Let's focus on problem-solving together. Here are some guiding prompts:")
                for prompt in relevant_prompts:
                    print("Text Bot:", prompt)
            else:
                print("No relevant prompts")

            text_response = get_text_response(model_text, user_input, relevant_prompts)
            print("Text Bot:", text_response)
            chat_history.append(("Text Bot", text_response))

        # Option to exit the loop
        exit_choice = input("Type 'exit' to quit, or press Enter to continue: ")
        if exit_choice.lower() == "exit":
            break

    # Print chat history
    print("Chat history:")
    for role, text in chat_history:
        print(f"{role}: {text}")

if __name__ == "__main__":
    main()


