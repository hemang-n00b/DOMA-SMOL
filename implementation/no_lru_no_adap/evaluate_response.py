import sys 
import os 
import openai

OPENAI_API_KEY = ""

def truncate_to_2048_words(text):
    words = text.split() 
    truncated_words = words[:2048]  
    return " ".join(truncated_words)

def evaluate_with_gpt4(question, generated_response, ideal_response):
    # Uses GPT-4 to evaluate the LLM-generated response on a Likert scale (1-5).
    ideal_response = truncate_to_2048_words(ideal_response)
    prompt = f"""
    Evaluate the following LLM-generated response on a Likert scale of 1-5 based on correctness, relevance, and fluency. Be a bit linient while scoring and give higher scores.
    1 = Poor, 2 = Fair, 3 = Good, 4 = Very Good, 5 = Excellent.
    
    **Question:** {question}
    **LLM Response:** {generated_response}
    **Ideal Response:** {ideal_response}

    Provide only the rating (a single number between 1 and 5) and no additional text.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    score = response.choices[0].message.content.strip()
    return int(score) if score.isdigit() else "Error in scoring"



if __name__ == "__main__":
    question = "What is the capital of France?"
    generated_response = "Paris is the capital of France."
    ideal_response = "The capital of France is Paris."

    score = evaluate_with_gpt4(question, generated_response, ideal_response)
    print(f"GPT-4 Likert Scale Score: {score}")
    
    print(score)
    
    
    
    