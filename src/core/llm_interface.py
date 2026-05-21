"""
LLM Interface Module

This module provides a unified interface for communicating with various
Language Model providers including:
- Ollama native chat API
- chat-completion compatible HTTP endpoints (via the `openai` Python client)

It handles prompt formatting, response generation, and error handling.
"""

import logging
from typing import Dict, List, Optional, Any
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False
    logging.warning("requests package not available")

# The openai Python client is imported lazily only when a chat-completion compatible provider is used.
OpenAI = None
OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Unified interface for various Language Model providers.
    """

    def __init__(self, config: Dict):
        """
        Initialize the LLM interface.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.llm_config = config["llm"]
        self.provider = self.llm_config["provider"]
        self.client = None
        self.last_error = ""
        
        # Initialize the appropriate client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on provider."""
        try:
            if self.provider == "ollama":
                if not REQUESTS_AVAILABLE:
                    raise ImportError("requests package required for Ollama provider")
                self.client = None
                self.llm_config["base_url"] = self.llm_config.get("base_url", "http://localhost:11434").rstrip("/")
                logger.info("Initialized Ollama LLM client")

            elif self.provider == "local" or self.provider == "openai":
                global OpenAI, OPENAI_AVAILABLE
                if OpenAI is None:
                    try:
                        from openai import OpenAI as _OpenAI

                        OpenAI = _OpenAI
                        OPENAI_AVAILABLE = True
                    except ImportError:
                        OPENAI_AVAILABLE = False

                if not OPENAI_AVAILABLE:
                    raise ImportError("openai package required")
                
                self.client = OpenAI(
                    base_url=self.llm_config["base_url"],
                    api_key=self.llm_config["api_key"]
                )
                logger.info(f"Initialized {self.provider} LLM client")
            
            else:
                logger.error(f"Unsupported LLM provider: {self.provider}")
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
            raise

    def generate_response(self, query: str, context: str = "", conversation_history: str = "", 
                         system_prompt: str = "") -> str:
        """
        Generate a response to a query with context.
        
        Args:
            query: User query
            context: Relevant document context
            conversation_history: Previous conversation context
            system_prompt: System prompt for the LLM
            
        Returns:
            Generated response string
        """
        try:
            self.last_error = ""
            # Prepare the prompt
            formatted_prompt = self._format_prompt(query, context, conversation_history)
            
            # Create messages
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": formatted_prompt})
            
            # Generate response
            response = self._call_llm(messages)
            
            return response
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error generating response: {e}")
            return (
                "I couldn't get a response from the configured LLM. "
                f"Last error: {self.last_error}"
            )

    def _format_prompt(self, query: str, context: str = "", conversation_history: str = "") -> str:
        """Format the prompt with query and context."""
        prompt_parts = []
        
        if conversation_history:
            prompt_parts.append(f"Previous conversation:\n{conversation_history}\n")
        
        if context:
            prompt_parts.append(f"Context information:\n{context}\n")
        
        prompt_parts.append(f"Question: {query}")
        
        return "\n".join(prompt_parts)

    def _call_llm(self, messages: List[Dict]) -> str:
        """Call the LLM with prepared messages."""
        try:
            if self.provider == "ollama":
                return self._call_ollama(messages)

            completion = self.client.chat.completions.create(
                model=self.llm_config["model"],
                messages=messages,
                temperature=self.llm_config.get("temperature", 0.7),
                max_tokens=self.llm_config.get("max_tokens", 2000),
                timeout=self.llm_config.get("timeout", 30)
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error calling LLM: {e}")
            raise

    def _call_ollama(self, messages: List[Dict]) -> str:
        """Call Ollama's native chat API."""
        response = requests.post(
            f"{self.llm_config['base_url']}/api/chat",
            json={
                "model": self.llm_config["model"],
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.llm_config.get("temperature", 0.7),
                    "num_predict": self.llm_config.get("max_tokens", 2000),
                    "num_ctx": self.llm_config.get("context_window", 8192),
                },
            },
            timeout=self.llm_config.get("timeout", 30),
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if content is None:
            raise ValueError("Ollama response did not include message.content")
        return content

    def check_relevance(self, query: str, summaries: Dict[str, str]) -> List[str]:
        """
        Check which documents are relevant to a query.
        
        Args:
            query: User query
            summaries: Dictionary of document IDs to summaries
            
        Returns:
            List of relevant document IDs
        """
        try:
            relevance_prompt = f"""
            Based on the following query and document summaries, return a JSON list of document IDs 
            that are relevant to answering the query.
            
            Query: {query}
            
            Document summaries:
            {json.dumps(summaries, indent=2)}
            
            Return only a JSON list like: ["id1", "id2", "id3"]
            Do not include any other text or explanation.
            """
            
            messages = [{"role": "user", "content": relevance_prompt}]
            response = self._call_llm(messages)
            
            try:
                relevant_ids = json.loads(response.strip())
                if isinstance(relevant_ids, list):
                    return relevant_ids
                else:
                    logger.warning("LLM returned non-list response for relevance check")
                    return []
            except json.JSONDecodeError:
                logger.warning("LLM returned invalid JSON for relevance check")
                return []
                
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return []

    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary string
        """
        try:
            prompt = f"""
            Summarize the following text in {max_length} characters or less:
            
            {text}
            
            Summary:
            """
            
            messages = [{"role": "user", "content": prompt}]
            summary = self._call_llm(messages)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        try:
            prompt = f"""
            Extract the {max_keywords} most important keywords from the following text.
            Return them as a JSON list of strings.
            
            Text: {text}
            
            Keywords (JSON list):
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = self._call_llm(messages)
            
            try:
                keywords = json.loads(response.strip())
                if isinstance(keywords, list):
                    return keywords[:max_keywords]
                else:
                    return []
            except json.JSONDecodeError:
                # Fallback: split response by commas
                return [kw.strip() for kw in response.split(',')[:max_keywords]]
                
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def classify_query(self, query: str, categories: List[str]) -> str:
        """
        Classify a query into one of the given categories.
        
        Args:
            query: Query to classify
            categories: List of possible categories
            
        Returns:
            Best matching category
        """
        try:
            prompt = f"""
            Classify the following query into one of these categories: {', '.join(categories)}
            
            Query: {query}
            
            Return only the category name, no explanation.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = self._call_llm(messages).strip()
            
            # Find the best matching category
            response_lower = response.lower()
            for category in categories:
                if category.lower() in response_lower:
                    return category
            
            # If no exact match, return the first category as default
            return categories[0] if categories else "general"
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return categories[0] if categories else "general"

    def generate_questions(self, text: str, num_questions: int = 3) -> List[str]:
        """
        Generate questions that can be answered by the given text.
        
        Args:
            text: Text to generate questions for
            num_questions: Number of questions to generate
            
        Returns:
            List of questions
        """
        try:
            prompt = f"""
            Generate {num_questions} questions that can be answered using the following text.
            Return the questions as a JSON list.
            
            Text: {text}
            
            Questions (JSON list):
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = self._call_llm(messages)
            
            try:
                questions = json.loads(response.strip())
                if isinstance(questions, list):
                    return questions[:num_questions]
                else:
                    return []
            except json.JSONDecodeError:
                # Fallback: split by line breaks and filter
                lines = response.strip().split('\n')
                questions = [line.strip() for line in lines if line.strip().endswith('?')]
                return questions[:num_questions]
                
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []

    def evaluate_answer(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate the quality of an answer given the question and context.
        
        Args:
            question: The original question
            answer: The generated answer
            context: The context used to generate the answer
            
        Returns:
            Evaluation results dictionary
        """
        try:
            prompt = f"""
            Evaluate the following answer based on the question and context provided.
            Rate the answer on a scale of 1-10 for accuracy, relevance, and completeness.
            Return a JSON object with ratings and brief explanations.
            
            Question: {question}
            Context: {context}
            Answer: {answer}
            
            Return JSON format:
            {{
                "accuracy": <1-10>,
                "relevance": <1-10>,
                "completeness": <1-10>,
                "overall": <1-10>,
                "explanation": "<brief explanation>"
            }}
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = self._call_llm(messages)
            
            try:
                evaluation = json.loads(response.strip())
                return evaluation
            except json.JSONDecodeError:
                return {
                    "accuracy": 5,
                    "relevance": 5,
                    "completeness": 5,
                    "overall": 5,
                    "explanation": "Unable to parse evaluation"
                }
                
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return {
                "accuracy": 0,
                "relevance": 0,
                "completeness": 0,
                "overall": 0,
                "explanation": f"Error during evaluation: {e}"
            }

    def is_available(self) -> bool:
        """Check if the LLM is available."""
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            self._call_llm(test_messages)
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.provider,
            "model": self.llm_config["model"],
            "base_url": self.llm_config["base_url"],
            "available": self.is_available()
        }
