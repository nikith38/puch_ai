import asyncio
import os
import json
import base64
import re
import io
from typing import Annotated, Optional, List, Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_QUIZ_API_KEY = os.environ.get("GEMINI_QUIZ_API_KEY", GEMINI_API_KEY)

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"

# Configure Google AI
genai.configure(api_key=GEMINI_API_KEY)

# --- Quiz Configuration matching your TS implementation ---
QUIZ_CONFIG = {
    "GENERATION": {
        "DEFAULT_QUESTION_COUNT": 5,
        "CHOICES_PER_QUESTION": 4,
        "MAX_QUESTION_COUNT": 10,
        "MIN_QUESTION_COUNT": 3
    },
    "API": {
        "MODEL": "gemini-1.5-flash",
        "CANDIDATE_COUNT": 1,
        "TEMPERATURE": 0.7,
        "TOP_P": 0.9,
        "TOP_K": 40,
        "MAX_TOKENS": 2048
    },
    "FEEDBACK": {
        "EXCELLENT": 90,
        "GREAT": 80,
        "GOOD": 70,
        "FAIR": 60,
        "NEEDS_WORK": 50
    },
    "ERRORS": {
        "INVALID_RESPONSE": "Invalid response from AI",
        "GENERATION_FAILED": "Failed to generate quiz",
        "PARSE_ERROR": "Failed to parse response"
    }
}

QUIZ_VALIDATION_RULES = {
    "TOPIC": {
        "MIN_LENGTH": 2,
        "MAX_LENGTH": 200
    },
    "QUESTIONS": {
        "MIN_COUNT": 3,
        "MAX_COUNT": 10
    }
}

# --- Data Models ---
class QuizChoice(BaseModel):
    id: str
    text: str

class QuizQuestion(BaseModel):
    id: str
    prompt: str
    choices: List[QuizChoice]
    correct_choice_id: str
    explanation: Optional[str] = None

class Quiz(BaseModel):
    topic: str
    questions: List[QuizQuestion]

class QuizResult(BaseModel):
    score: int
    total: int
    percentage: int
    correct_answers: List[str]
    incorrect_answers: List[str]

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: Optional[str] = None

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="cat-tutor-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Cat Tutor AI Class (using gemini-2.0-flash-exp like your TS) ---
class CatTutorAI:
    def __init__(self):
        # Use the same model as your TS implementation for image generation
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.additional_instructions = """
        You are a cat tutor! Use a fun story about lots of tiny cats as a metaphor.
        Keep sentences short but conversational, casual, and engaging.
        Generate a cute, minimal illustration for each sentence with black ink on white background.
        No commentary, just begin your explanation.
        Keep going until you're done.
        Generate actual images along with your explanations - don't just describe them.
        """

    async def explain_topic_with_images(self, topic: str) -> List[TextContent | ImageContent]:
        """
        Explain topic with cat metaphors and generate actual images
        """
        try:
            prompt = f"""
            {self.additional_instructions}
            
            Topic to explain: {topic}
            
            Please explain this topic using cat metaphors. For each key concept, generate a cute cat doodle illustration.
            The illustrations should be minimal, black ink on white background, featuring cute cats.
            Mix text explanations with images throughout your response.
            """
            
            # Generate content with both text and images
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    candidate_count=1,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=2048,
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            result = []
            
            # Check if response has multiple parts (text + images)
            if hasattr(response, 'parts') and response.parts:
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        result.append(TextContent(type="text", text=part.text))
                    elif hasattr(part, 'inline_data'):
                        # Convert inline image data to ImageContent
                        image_data = part.inline_data
                        result.append(ImageContent(
                            type="image",
                            mimeType=image_data.mime_type,
                            data=image_data.data
                        ))
            else:
                # Fallback to text only if no images generated
                text_content = response.text if hasattr(response, 'text') else str(response)
                result.append(TextContent(type="text", text=text_content))
            
            # Add quiz offer at the end
            result.append(TextContent(
                type="text", 
                text="\n\n🐱 **Meow!** Do you want to have a short quiz about this topic? Just ask me to 'create a quiz'!"
            ))
            
            return result
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to explain topic: {str(e)}"))

# --- Quiz Handler Class (matching your TS implementation exactly) ---
class QuizHandler:
    def __init__(self):
        # Use separate API key for quiz operations like your TS implementation
        quiz_api_key = GEMINI_QUIZ_API_KEY
        if not quiz_api_key or quiz_api_key == 'your_gemini_quiz_api_key_here':
            raise Exception("""
❌ Gemini Quiz API key not configured properly!

Please follow these steps:
1. Get a free API key from: https://aistudio.google.com/app/apikey
2. Edit the .env file in your project root
3. Add: GEMINI_QUIZ_API_KEY=your_actual_api_key_here
4. Restart the development server

The API key should look like: AIzaSyC...
            """)
        
        # Configure separate AI client for quiz
        genai.configure(api_key=quiz_api_key)
        self.model = genai.GenerativeModel(QUIZ_CONFIG["API"]["MODEL"])
    
    def validate_input(self, topic: str, question_count: int) -> None:
        """Validate input parameters"""
        if not topic or len(topic) < QUIZ_VALIDATION_RULES["TOPIC"]["MIN_LENGTH"] or len(topic) > QUIZ_VALIDATION_RULES["TOPIC"]["MAX_LENGTH"]:
            raise ValueError(f"Topic must be between {QUIZ_VALIDATION_RULES['TOPIC']['MIN_LENGTH']} and {QUIZ_VALIDATION_RULES['TOPIC']['MAX_LENGTH']} characters")
        
        if question_count < QUIZ_VALIDATION_RULES["QUESTIONS"]["MIN_COUNT"] or question_count > QUIZ_VALIDATION_RULES["QUESTIONS"]["MAX_COUNT"]:
            raise ValueError(f"Question count must be between {QUIZ_VALIDATION_RULES['QUESTIONS']['MIN_COUNT']} and {QUIZ_VALIDATION_RULES['QUESTIONS']['MAX_COUNT']}")

    async def generate_quiz(self, topic: str, question_count: int = QUIZ_CONFIG["GENERATION"]["DEFAULT_QUESTION_COUNT"], slide_content: Optional[str] = None) -> Quiz:
        """Generate a quiz based on a topic and optional slide content"""
        try:
            # Validate input
            self.validate_input(topic, question_count)
            
            prompt = self._build_quiz_prompt(topic, question_count, slide_content)
            
            # Use generateContent for more reliable responses (matching your TS implementation)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    candidate_count=QUIZ_CONFIG["API"]["CANDIDATE_COUNT"],
                    temperature=QUIZ_CONFIG["API"]["TEMPERATURE"],
                    top_p=QUIZ_CONFIG["API"]["TOP_P"],
                    top_k=QUIZ_CONFIG["API"]["TOP_K"],
                    max_output_tokens=QUIZ_CONFIG["API"]["MAX_TOKENS"],
                )
            )
            
            text = response.text.strip() if hasattr(response, 'text') and response.text else ''
            if not text:
                raise ValueError(QUIZ_CONFIG["ERRORS"]["INVALID_RESPONSE"])
            
            quiz_data = self._extract_quiz_from_response(text)
            return Quiz(**quiz_data)
            
        except Exception as e:
            print(f'Quiz generation error: {e}')
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"{QUIZ_CONFIG['ERRORS']['GENERATION_FAILED']}: {str(e)}"))
    
    def _build_quiz_prompt(self, topic: str, question_count: int, slide_content: Optional[str] = None) -> str:
        """Build quiz generation prompt"""
        if slide_content:
            prompt = f"""
            Create a quiz about "{topic}" with exactly {question_count} questions based on the following content:
            
            {slide_content}
            
            Each question should have exactly 4 choices (a, b, c, d).
            Include explanations for correct answers.
            Make it fun and engaging with cat-themed language where appropriate!
            
            Return the response in this exact JSON format:
            {{
                "topic": "{topic}",
                "questions": [
                    {{
                        "id": "q1",
                        "prompt": "Question text here",
                        "choices": [
                            {{"id": "a", "text": "Choice A"}},
                            {{"id": "b", "text": "Choice B"}},
                            {{"id": "c", "text": "Choice C"}},
                            {{"id": "d", "text": "Choice D"}}
                        ],
                        "correct_choice_id": "a",
                        "explanation": "Explanation of why this is correct"
                    }}
                ]
            }}
            """
        else:
            prompt = f"""
            Create a multiple choice quiz about "{topic}" with exactly {question_count} questions.
            Each question should have exactly 4 choices (a, b, c, d).
            Include explanations for correct answers.
            Make it fun and engaging with cat-themed language where appropriate!
            
            Return the response in this exact JSON format:
            {{
                "topic": "{topic}",
                "questions": [
                    {{
                        "id": "q1",
                        "prompt": "Question text here",
                        "choices": [
                            {{"id": "a", "text": "Choice A"}},
                            {{"id": "b", "text": "Choice B"}},
                            {{"id": "c", "text": "Choice C"}},
                            {{"id": "d", "text": "Choice D"}}
                        ],
                        "correct_choice_id": "a",
                        "explanation": "Explanation of why this is correct"
                    }}
                ]
            }}
            """
        
        return prompt
    
    def _extract_quiz_from_response(self, text: str) -> Dict[str, Any]:
        """Extract quiz data from AI response (matching your TS implementation)"""
        try:
            # Clean the response text
            cleaned = text.strip()
            
            # Remove markdown code blocks if present
            code_block_regex = r'```(?:json)?\s*([\s\S]*?)\s*```'
            match = re.search(code_block_regex, cleaned, re.IGNORECASE)
            if match:
                cleaned = match.group(1)
            
            # Remove leading "json" labels
            cleaned = re.sub(r'^json\s*', '', cleaned, flags=re.IGNORECASE).strip()
            
            # Find the first complete JSON object
            start_index = cleaned.find('{')
            end_index = cleaned.rfind('}')
            
            if start_index == -1 or end_index == -1:
                raise ValueError('No valid JSON object found in response')
            
            json_text = cleaned[start_index:end_index + 1]
            
            # Normalize quotes and parse
            normalized = re.sub(r'["""]', '"', json_text)
            quiz_data = json.loads(normalized)
            
            # Validate quiz structure
            self._validate_quiz_structure(quiz_data)
            
            return quiz_data
            
        except Exception as e:
            raise ValueError(f"{QUIZ_CONFIG['ERRORS']['PARSE_ERROR']}: {str(e)}")
    
    def _validate_quiz_structure(self, data: Any) -> None:
        """Validate the quiz structure (matching your TS implementation)"""
        if not data.get("topic") or not isinstance(data["topic"], str):
            raise ValueError('Invalid quiz: missing or invalid topic')
        
        if not isinstance(data.get("questions"), list) or not data["questions"]:
            raise ValueError('Invalid quiz: missing or empty questions array')
        
        for index, question in enumerate(data["questions"]):
            if not question.get("id") or not question.get("prompt") or not isinstance(question.get("choices"), list):
                raise ValueError(f'Invalid question {index + 1}: missing required fields')
            
            if len(question["choices"]) != QUIZ_CONFIG["GENERATION"]["CHOICES_PER_QUESTION"]:
                raise ValueError(f'Question {index + 1}: must have exactly {QUIZ_CONFIG["GENERATION"]["CHOICES_PER_QUESTION"]} choices')
            
            if not question.get("correct_choice_id"):
                raise ValueError(f'Question {index + 1}: missing correct answer')
            
            # Validate choices
            for choice_index, choice in enumerate(question["choices"]):
                if not choice.get("id") or not choice.get("text"):
                    raise ValueError(f'Question {index + 1}, choice {choice_index + 1}: missing id or text')

    def calculate_results(self, quiz: Quiz, user_answers: Dict[str, str]) -> QuizResult:
        """Calculate quiz results (matching your TS implementation)"""
        score = 0
        correct_answers = []
        incorrect_answers = []
        
        for question in quiz.questions:
            user_answer = user_answers.get(question.id)
            if user_answer == question.correct_choice_id:
                score += 1
                correct_answers.append(question.id)
            else:
                incorrect_answers.append(question.id)
        
        total = len(quiz.questions)
        percentage = round((score / total) * 100)
        
        return QuizResult(
            score=score,
            total=total,
            percentage=percentage,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers
        )
    
    def get_performance_feedback(self, score: int, total: int) -> Dict[str, str]:
        """Get performance feedback based on score (matching your TS implementation)"""
        percentage = (score / total) * 100
        
        if percentage >= QUIZ_CONFIG["FEEDBACK"]["EXCELLENT"]:
            return {"emoji": "😺", "message": "Purr‑fect!", "color": "#28a745"}
        elif percentage >= QUIZ_CONFIG["FEEDBACK"]["GREAT"]:
            return {"emoji": "😸", "message": "Meow‑gnificent!", "color": "#20c997"}
        elif percentage >= QUIZ_CONFIG["FEEDBACK"]["GOOD"]:
            return {"emoji": "🙂", "message": "Good whiskers!", "color": "#17a2b8"}
        elif percentage >= QUIZ_CONFIG["FEEDBACK"]["FAIR"]:
            return {"emoji": "😼", "message": "Not bad, keep practicing!", "color": "#ffc107"}
        elif percentage >= QUIZ_CONFIG["FEEDBACK"]["NEEDS_WORK"]:
            return {"emoji": "😿", "message": "More cat naps (study) needed!", "color": "#fd7e14"}
        else:
            return {"emoji": "😭", "message": "Time for intensive cat training!", "color": "#dc3545"}

    def get_detailed_feedback(self, quiz: Quiz, incorrect_answers: List[str]) -> List[Dict[str, str]]:
        """Get detailed feedback for incorrect answers (matching your TS implementation)"""
        return [
            {
                "question_id": question_id,
                "explanation": next(
                    (q.explanation or "No explanation available for this question." 
                     for q in quiz.questions if q.id == question_id), 
                    "Question not found."
                )
            }
            for question_id in incorrect_answers
        ]

    def is_configured(self) -> bool:
        """Check if the quiz module is properly configured"""
        return bool(GEMINI_QUIZ_API_KEY and GEMINI_QUIZ_API_KEY != 'your_gemini_quiz_api_key_here')

    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration status"""
        return {
            "configured": self.is_configured(),
            "api_key_present": bool(GEMINI_QUIZ_API_KEY and GEMINI_QUIZ_API_KEY != 'your_gemini_quiz_api_key_here'),
            "model": QUIZ_CONFIG["API"]["MODEL"]
        }

# --- Initialize services ---
cat_tutor = CatTutorAI()
quiz_handler = QuizHandler()

# Store quizzes in memory for answer checking (in production, use proper storage)
quiz_storage: Dict[str, Quiz] = {}

# --- MCP Server Setup ---
mcp = FastMCP(
    "Cat Tutor MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    """Validation tool required by Puch AI"""
    return MY_NUMBER

# --- Tool: explain_topic ---
EXPLAIN_TOPIC_DESCRIPTION = RichToolDescription(
    description="Explain any topic using fun cat stories and generate cute cat doodle illustrations",
    use_when="When user asks 'explain me [topic]' or requests topic explanation with visual learning",
    side_effects="Returns educational content with cat metaphors and actual generated cat doodle images"
)

@mcp.tool(description=EXPLAIN_TOPIC_DESCRIPTION.model_dump_json())
async def explain_topic(
    topic: Annotated[str, Field(description="The topic to explain using cat metaphors and illustrations")]
) -> list[TextContent | ImageContent]:
    """
    Explains a topic using cat stories and generates actual cat doodle images.
    Matches the behavior of your TypeScript implementation.
    """
    try:
        # Generate explanation with images using the same model as your TS code
        content_list = await cat_tutor.explain_topic_with_images(topic)
        return content_list
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to explain topic: {str(e)}"))

# --- Tool: create_quiz ---
CREATE_QUIZ_DESCRIPTION = RichToolDescription(
    description="Generate a multiple choice quiz with cat-themed questions and encouragement",
    use_when="When user wants to test their knowledge after learning a topic or requests a quiz",
    side_effects="Creates interactive quiz questions with explanations and stores quiz for answer checking"
)

@mcp.tool(description=CREATE_QUIZ_DESCRIPTION.model_dump_json())
async def create_quiz(
    topic: Annotated[str, Field(description="Topic for the quiz")],
    question_count: Annotated[int, Field(description="Number of questions (3-10)")] = 5,
    slide_content: Annotated[Optional[str], Field(description="Optional content to base quiz on")] = None
) -> str:
    """
    Creates a multiple choice quiz about the specified topic.
    Matches the functionality of your TypeScript QuizHandler.
    """
    try:
        # Generate quiz using the same logic as your TS implementation
        quiz = await quiz_handler.generate_quiz(topic, question_count, slide_content)
        
        # Store quiz for later answer checking
        quiz_id = f"{topic.lower().replace(' ', '_')}_quiz_{len(quiz_storage)}"
        quiz_storage[quiz_id] = quiz
        
        # Format quiz for display (matching your frontend expectations)
        quiz_text = f"🐱 **Cat Quiz: {quiz.topic}**\n\n"
        
        for i, question in enumerate(quiz.questions, 1):
            quiz_text += f"**Question {i}:** {question.prompt}\n\n"
            
            for choice in question.choices:
                quiz_text += f"   {choice.id.upper()}) {choice.text}\n"
            
            quiz_text += "\n"
        
        quiz_text += f"📝 *Send me your answers like: 'quiz_id={quiz_id}&answers=a,b,c,d,a' to get your results!*\n"
        quiz_text += f"🎯 *Quiz ID: {quiz_id}*"
        
        return quiz_text
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to create quiz: {str(e)}"))

# --- Tool: check_quiz_answers ---
CHECK_ANSWERS_DESCRIPTION = RichToolDescription(
    description="Check quiz answers and provide detailed cat-themed feedback with performance analysis",
    use_when="When user submits answers to a quiz they've taken",
    side_effects="Returns score, detailed feedback, explanations with cat encouragement, and performance analysis"
)

@mcp.tool(description=CHECK_ANSWERS_DESCRIPTION.model_dump_json())
async def check_quiz_answers(
    quiz_id: Annotated[str, Field(description="Quiz ID from the created quiz")],
    user_answers: Annotated[str, Field(description="User answers as comma-separated values (e.g., 'a,b,c,d,a')")],
) -> str:
    """
    Check user's quiz answers and provide feedback with cat-themed encouragement.
    Matches the functionality of your TypeScript QuizHandler.calculateResults.
    """
    try:
        # Retrieve quiz from storage
        if quiz_id not in quiz_storage:
            return f"🙀 Oops! Quiz '{quiz_id}' not found. Please create a quiz first!"
        
        quiz = quiz_storage[quiz_id]
        
        # Parse user answers
        answer_list = [answer.strip().lower() for answer in user_answers.split(',')]
        
        if len(answer_list) != len(quiz.questions):
            return f"🙀 Oops! Expected {len(quiz.questions)} answers but got {len(answer_list)}. Please try again!"
        
        # Convert to dict format for calculation
        user_answer_dict = {
            question.id: answer_list[i] 
            for i, question in enumerate(quiz.questions)
        }
        
        # Calculate results using the same logic as your TS implementation
        results = quiz_handler.calculate_results(quiz, user_answer_dict)
        feedback = quiz_handler.get_performance_feedback(results.score, results.total)
        
        # Format response matching your TS feedback style
        response = f"""🎉 **Quiz Results for {quiz.topic}** 🎉

{feedback['emoji']} **Score: {results.score}/{results.total} ({results.percentage}%)**

{feedback['message']}

📊 **Detailed Results:**
✅ Correct: {len(results.correct_answers)} questions
❌ Incorrect: {len(results.incorrect_answers)} questions

"""
        
        # Add detailed feedback for incorrect answers (matching your TS implementation)
        if results.incorrect_answers:
            response += "📚 **Let's Review:**\n\n"
            detailed_feedback = quiz_handler.get_detailed_feedback(quiz, results.incorrect_answers)
            
            for feedback_item in detailed_feedback:
                question = next(q for q in quiz.questions if q.id == feedback_item["question_id"])
                correct_choice = next(c for c in question.choices if c.id == question.correct_choice_id)
                
                response += f"**{question.prompt}**\n"
                response += f"Correct answer: {question.correct_choice_id.upper()}) {correct_choice.text}\n"
                response += f"💡 {feedback_item['explanation']}\n\n"
        
        # Add encouraging message based on performance (matching your TS style)
        if results.percentage >= 80:
            response += "\n🌟 Excellent work! You're a true cat scholar! 🐾"
        elif results.percentage >= 60:
            response += "\n😸 Good job! Keep up the great work! 🐱"
        else:
            response += "\n😿 Don't worry, every cat needs practice! Try reviewing the topic again. 🐱📚"
        
        return response
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to check answers: {str(e)}"))

# --- Tool: generate_encouragement_cat ---
ENCOURAGEMENT_DESCRIPTION = RichToolDescription(
    description="Generate encouraging cat doodle images for quiz interactions and learning moments",
    use_when="To provide visual encouragement during quiz, after answering questions, or during learning",
    side_effects="Returns a cute cat doodle image with encouraging message"
)

@mcp.tool(description=ENCOURAGEMENT_DESCRIPTION.model_dump_json())
async def generate_encouragement_cat(
    message_type: Annotated[str, Field(description="Type of encouragement: 'good_job', 'try_again', 'excellent', 'start_quiz', 'keep_going'")] = "good_job",
    custom_message: Annotated[Optional[str], Field(description="Custom encouraging message to include with the cat")] = None
) -> list[TextContent | ImageContent]:
    """
    Generate encouraging cat images for quiz interactions and learning.
    Uses the same image generation model as your main explanation tool.
    """
    try:
        # Create prompts for different encouragement types
        prompts = {
            "good_job": "Generate a cute cat doodle giving thumbs up with sparkles around it, black ink on white background, minimal style",
            "try_again": "Generate a supportive cat doodle with encouraging paws up, surrounded by hearts, minimal black ink on white background",
            "excellent": "Generate a cat doodle wearing a graduation cap with stars and confetti, celebration style, black ink on white background",
            "start_quiz": "Generate an excited cat doodle with a pencil and paper, ready to learn, simple black lines on white background",
            "keep_going": "Generate a determined cat doodle with a superhero cape, motivational pose, minimal black ink style"
        }
        
        prompt = prompts.get(message_type, prompts["good_job"])
        if custom_message:
            prompt += f". Include visual elements that represent: {custom_message}"
        
        # Generate image using the same model as topic explanations
        response = cat_tutor.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                candidate_count=1,
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                max_output_tokens=1024,
            )
        )
        
        result = []
        
        # Process response for images (same logic as explain_topic)
        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    result.append(TextContent(type="text", text=part.text))
                elif hasattr(part, 'inline_data'):
                    image_data = part.inline_data
                    result.append(ImageContent(
                        type="image",
                        mimeType=image_data.mime_type,
                        data=image_data.data
                    ))
        
        # Add encouraging text message
        messages = {
            "good_job": "🎉 Great job! You're doing pawsome!",
            "try_again": "💪 Don't give up! Every great cat started as a kitten!",
            "excellent": "🌟 Absolutely purr-fect! You're a star student!",
            "start_quiz": "📚 Ready to test your knowledge? Let's go!",
            "keep_going": "🚀 You've got this! Keep up the great work!"
        }
        
        encouragement_text = custom_message or messages.get(message_type, messages["good_job"])
        result.append(TextContent(type="text", text=encouragement_text))
        
        return result
        
    except Exception as e:
        # Fallback to text-only encouragement if image generation fails
        fallback_messages = {
            "good_job": "🐱 Great job! You're doing pawsome! 🎉",
            "try_again": "😸 Don't give up! Every great cat started as a kitten! 💪",
            "excellent": "😺 Absolutely purr-fect! You're a star student! 🌟",
            "start_quiz": "🙀 Ready to test your knowledge? Let's go! 📚",
            "keep_going": "😻 You've got this! Keep up the great work! 🚀"
        }
        
        message = custom_message or fallback_messages.get(message_type, fallback_messages["good_job"])
        return [TextContent(type="text", text=f"{message}\n\n(Image generation temporarily unavailable, but the encouragement is real! 🐾)")]

# --- Tool: get_quiz_status ---
QUIZ_STATUS_DESCRIPTION = RichToolDescription(
    description="Get the configuration status of the quiz system and list available quizzes",
    use_when="To check if quiz system is properly configured or see available quizzes",
    side_effects="Returns system status and quiz availability information"
)

@mcp.tool(description=QUIZ_STATUS_DESCRIPTION.model_dump_json())
async def get_quiz_status() -> str:
    """
    Get quiz system status and available quizzes.
    Matches the getConfigStatus functionality from your TS implementation.
    """
    try:
        # Get configuration status
        config_status = quiz_handler.get_config_status()
        
        status_text = f"""🐱 **Cat Tutor Quiz System Status**

🔧 **Configuration:**
- ✅ Configured: {config_status['configured']}
- 🔑 API Key Present: {config_status['api_key_present']}
- 🤖 Model: {config_status['model']}

📚 **Available Quizzes:** {len(quiz_storage)}
"""
        
        if quiz_storage:
            status_text += "\n🎯 **Quiz List:**\n"
            for quiz_id, quiz in quiz_storage.items():
                status_text += f"- {quiz_id}: {quiz.topic} ({len(quiz.questions)} questions)\n"
        else:
            status_text += "\n📝 No quizzes created yet. Create your first quiz by asking me to explain a topic!"
        
        status_text += f"""
🎨 **Image Generation:** Using {cat_tutor.model.model_name}
🧠 **Quiz Generation:** Using {quiz_handler.model.model_name}

💡 **Quick Start:**
1. Ask me to "explain photosynthesis" (or any topic)
2. I'll explain with cute cat stories and images
3. Then ask to "create a quiz about photosynthesis"  
4. Take the quiz and get feedback!
"""
        
        return status_text
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to get quiz status: {str(e)}"))

# --- Tool: clear_quiz_storage ---
CLEAR_STORAGE_DESCRIPTION = RichToolDescription(
    description="Clear all stored quizzes from memory (admin function)",
    use_when="To clear quiz storage for testing or maintenance",
    side_effects="Removes all stored quizzes from memory"
)

@mcp.tool(description=CLEAR_STORAGE_DESCRIPTION.model_dump_json())
async def clear_quiz_storage() -> str:
    """
    Clear all stored quizzes from memory.
    Useful for testing and maintenance.
    """
    try:
        cleared_count = len(quiz_storage)
        quiz_storage.clear()
        
        return f"🧹 Cleared {cleared_count} quizzes from storage. Ready for fresh cat adventures!"
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to clear storage: {str(e)}"))

# --- Run MCP Server ---
async def main():
    print("🐱 Starting Cat Tutor MCP Server on http://0.0.0.0:8086")
    print("🎓 Ready to explain topics with cats and create fun quizzes!")
    print(f"🔑 Using Gemini API Key: {GEMINI_API_KEY[:20]}..." if GEMINI_API_KEY else "❌ No API key configured")
    print(f"🎯 Quiz API Key: {GEMINI_QUIZ_API_KEY[:20]}..." if GEMINI_QUIZ_API_KEY else "❌ No quiz API key configured")
    print("")
    print("🚀 **Available Tools:**")
    print("   📚 explain_topic - Explain topics with cat stories and images")
    print("   🎯 create_quiz - Generate cat-themed quizzes")
    print("   ✅ check_quiz_answers - Grade quizzes with encouraging feedback")
    print("   😺 generate_encouragement_cat - Create encouraging cat images")
    print("   📊 get_quiz_status - Check system status")
    print("   🧹 clear_quiz_storage - Clear stored quizzes")
    print("   ✓ validate - Puch AI validation")
    print("")
    
    try:
        await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)
    except KeyboardInterrupt:
        print("\n👋 Cat Tutor MCP Server shutting down. Goodbye!")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())