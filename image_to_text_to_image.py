import os
import sys
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please add your API key to the .env file."
    )

client = OpenAI(api_key=API_KEY)


# Default input and output files
DEFAULT_INPUT = "messi.jpeg"
DEFAULT_OUTPUT = "messi_generated.png"


def encode_image(image_path):
    """Convert the image into Base64 format."""

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_image(image_path):
    """Send the image to a vision-capable model and get a description."""

    image_base64 = encode_image(image_path)

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analyze this image carefully and create a detailed "
                            "visual description that can be used as a prompt for "
                            "an image-generation model. Describe the main subject, "
                            "people, clothing, objects, environment, background, "
                            "colors, lighting, composition, camera angle, "
                            "perspective, facial expressions, poses, textures, "
                            "and other important visual details. "
                            "Do not make assumptions that cannot be visually "
                            "supported by the image."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:image/jpeg;base64,{image_base64}"
                        ),
                    },
                ],
            }
        ],
    )

    return response.output_text


def generate_image(description, output_path):
    """Generate a new image using the textual description."""

    print("\nGenerating new image...")

    result = client.images.generate(
        model="gpt-image-2",
        prompt=description,
    )

    image_base64 = result.data[0].b64_json

    image_bytes = base64.b64decode(image_base64)

    with open(output_path, "wb") as output_file:
        output_file.write(image_bytes)


def main():

    # Allow optional command-line arguments
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    print("=" * 50)
    print("Exercise #7 — Image → Text → Image")
    print("=" * 50)

    # Check that the input image exists
    if not Path(input_path).exists():
        print(f"\nError: Image '{input_path}' was not found.")
        print("Make sure the image is in the same folder as this program.")
        return

    try:

        # ------------------------------------------------
        # STEP 1 — Read the image
        # ------------------------------------------------

        print(f"\n[1/4] Reading image: {input_path}")

        # ------------------------------------------------
        # STEP 2 — Image → Text
        # ------------------------------------------------

        print("[2/4] Sending image to vision model...")

        description = describe_image(input_path)

        print("\n" + "=" * 50)
        print("GENERATED IMAGE DESCRIPTION")
        print("=" * 50)

        print(description)

        print("=" * 50)

        # ------------------------------------------------
        # STEP 3 — Text → Image
        # ------------------------------------------------

        print("\n[3/4] Sending description to image-generation model...")

        generate_image(description, output_path)

        # ------------------------------------------------
        # STEP 4 — Save image
        # ------------------------------------------------

        print(f"[4/4] Generated image saved as: {output_path}")

        print("\nPipeline completed successfully!")

        print("\nPipeline:")
        print("messi.jpeg")
        print("    ↓")
        print("Vision model")
        print("    ↓")
        print("Text description")
        print("    ↓")
        print("Image generation model")
        print("    ↓")
        print(output_path)

    except Exception as error:

        print("\nERROR")
        print("-" * 50)
        print(error)
        print("-" * 50)


if __name__ == "__main__":
    main()