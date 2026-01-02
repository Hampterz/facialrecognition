"""
Google Gemini API integration for chat responses.
"""

# Lazy import to prevent startup crashes
genai = None

def _import_genai():
    """Lazy import of google.generativeai."""
    global genai
    if genai is None:
        try:
            import google.generativeai as gai
            genai = gai
            return True
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Install with: pip install google-generativeai"
            )
    return True


def initialize_gemini(api_key: str):
    """
    Initialize Gemini API with API key.
    
    Args:
        api_key: Google Gemini API key
    """
    try:
        _import_genai()
        genai.configure(api_key=api_key)
        print("✓ Gemini API initialized successfully!")
        return True
    except ImportError as e:
        print(f"Error: {str(e)}")
        return False
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        return False


def get_gemini_response(prompt: str, api_key: str):
    """
    Get response from Gemini API.
    
    Args:
        prompt: User's text prompt
        api_key: Google Gemini API key
        
    Returns:
        Gemini's response text, or error message
    """
    try:
        _import_genai()
        
        if not api_key or api_key.strip() == "":
            return "Error: Gemini API key not set. Please enter your API key in Settings."
        
        genai.configure(api_key=api_key)
        
        # Use Gemini Pro model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "No response from Gemini."
            
    except ImportError as e:
        return f"Error: google-generativeai package not installed. Install with: pip install google-generativeai"
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "api key" in error_msg.lower():
            return "Error: Invalid Gemini API key. Please check your API key in Settings."
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return "Error: API quota exceeded. Please check your Gemini API usage limits."
        else:
            return f"Error: {error_msg}"

