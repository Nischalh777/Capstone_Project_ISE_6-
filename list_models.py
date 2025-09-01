import os
from dotenv import load_dotenv
import google.generativeai as genai

print("--- Attempting to connect to Google AI to list available models ---")

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("🔴 FATAL ERROR: Could not find GEMINI_API_KEY in your .env file.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        print("\n✅ Successfully configured API. Now fetching models...\n")
        
        print("--- Models available to your API key that support 'generateContent' ---")
        
        found_models = False
        for m in genai.list_models():
          if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
            found_models = True
            
        if not found_models:
            print("🔴 ERROR: No models supporting 'generateContent' were found for your API key.")
        else:
            print("\n----------------------------------------------------------------------")
            print("🟢 SUCCESS: Copy one of the model names listed above into your app.py file.")
            print("   For example: gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')")
            print("----------------------------------------------------------------------")

    except Exception as e:
        print(f"\n🔴 FATAL ERROR: An exception occurred during the process.")
        print(f"   ERROR DETAILS: {e}")
        print("\n   This confirms the issue is with your API key or environment setup.")