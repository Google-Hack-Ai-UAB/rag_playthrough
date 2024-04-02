def extract_generated_text(response):
    generated_text = response["candidates"][0]["content"]["parts"][0]["text"]
    return generated_text

if __name__ == "__main__":
    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "- C++\n- Neural Networks\n- Pytorch\n- Tensorflow\n..."
                        }
                    ]
                },
                "finish_reason": "STOP",
                "safety_ratings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    },
                ]
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 445,
            "candidates_token_count": 142,
            "total_token_count": 587
        }
    }

    generated_text = extract_generated_text(response)
    print(generated_text)
