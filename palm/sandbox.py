
import vertexai
from vertexai.language_models import TextGenerationModel

vertexai.init(project="deception-emotion", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison")
response = model.predict(
    """Repeat after me: hello world!""",
    **parameters
)
print(f"Response from Model: {response.text}")

# save model outputs into running pandas df, storing all parameters of the generation
