"""
Query Analyzer Module

This module provides intelligent query analysis and preprocessing including:
- Query intent classification
- Entity extraction
- Query expansion and refinement
- Context-aware query processing
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes and preprocesses user queries for better retrieval and response generation.
    """

    def __init__(self, config: Dict, llm_interface=None):
        """
        Initialize the query analyzer.
        
        Args:
            config: System configuration dictionary
            llm_interface: LLM interface for advanced analysis
        """
        self.config = config
        self.llm_interface = llm_interface
        
        # Query processing patterns
        self.question_patterns = [
            r'\bwhat\b', r'\bhow\b', r'\bwhen\b', r'\bwhere\b', r'\bwhy\b', r'\bwho\b',
            r'\bwhich\b', r'\bcan\b', r'\bis\b', r'\bare\b', r'\bdoes\b', r'\bdo\b'
        ]
        
        self.command_patterns = [
            r'\bshow\b', r'\blist\b', r'\bfind\b', r'\bsearch\b', r'\bget\b',
            r'\btell\b', r'\bexplain\b', r'\bdescribe\b', r'\bcompare\b'
        ]
        
        # Intent categories
        self.intent_categories = [
            "question", "command", "request", "comparison", "definition", "tutorial", "troubleshooting"
        ]

    def analyze_query(self, query: str, user_role: str = "User") -> str:
        """
        Analyze and preprocess a user query.
        
        Args:
            query: Raw user query
            user_role: Role of the user making the query
            
        Returns:
            Processed query string
        """
        try:
            # Basic preprocessing
            processed_query = self._preprocess_query(query)
            
            # Extract analysis components
            analysis = {
                "original_query": query,
                "processed_query": processed_query,
                "intent": self._classify_intent(processed_query),
                "entities": self._extract_entities(processed_query),
                "query_type": self._determine_query_type(processed_query),
                "user_role": user_role,
                "timestamp": datetime.now().isoformat()
            }
            
            # Enhanced processing based on analysis
            enhanced_query = self._enhance_query(analysis)
            
            logger.info(f"Query analyzed - Intent: {analysis['intent']}, Type: {analysis['query_type']}")
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return query  # Return original query if analysis fails

    def _preprocess_query(self, query: str) -> str:
        """Basic query preprocessing."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Handle common abbreviations
        abbreviations = {
            r'\bai\b': 'artificial intelligence',
            r'\bml\b': 'machine learning',
            r'\bdl\b': 'deep learning',
            r'\bnlp\b': 'natural language processing',
            r'\bapi\b': 'application programming interface'
        }
        
        for abbrev, full_form in abbreviations.items():
            query = re.sub(abbrev, full_form, query, flags=re.IGNORECASE)
        
        return query

    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the query."""
        query_lower = query.lower()
        
        # Check for question patterns
        if any(re.search(pattern, query_lower) for pattern in self.question_patterns):
            if 'how' in query_lower:
                return 'tutorial'
            elif 'what' in query_lower and ('is' in query_lower or 'are' in query_lower):
                return 'definition'
            elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
                return 'comparison'
            elif any(word in query_lower for word in ['problem', 'error', 'issue', 'fix', 'solve']):
                return 'troubleshooting'
            else:
                return 'question'
        
        # Check for command patterns
        elif any(re.search(pattern, query_lower) for pattern in self.command_patterns):
            return 'command'
        
        # Check for request patterns
        elif any(word in query_lower for word in ['please', 'can you', 'could you', 'help']):
            return 'request'
        
        else:
            return 'question'  # Default to question

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities and important terms from the query."""
        entities = []
        
        # Simple pattern-based entity extraction
        # Technical terms (capitalized words, acronyms)
        tech_terms = re.findall(r'\b[A-Z][A-Za-z]*\b', query)
        entities.extend(tech_terms)
        
        # Quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        # Numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities.extend(numbers)
        
        # Remove duplicates and filter short terms
        entities = list(set([entity for entity in entities if len(entity) > 2]))
        
        return entities

    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['list', 'show all', 'enumerate']):
            return 'list'
        elif any(word in query_lower for word in ['explain', 'describe', 'tell me about']):
            return 'explanation'
        elif any(word in query_lower for word in ['example', 'sample', 'demo']):
            return 'example'
        elif any(word in query_lower for word in ['how to', 'step by step', 'tutorial']):
            return 'tutorial'
        elif any(word in query_lower for word in ['compare', 'difference', 'vs']):
            return 'comparison'
        elif query_lower.endswith('?'):
            return 'question'
        else:
            return 'general'

    def _enhance_query(self, analysis: Dict) -> str:
        """Enhance the query based on analysis results."""
        query = analysis["processed_query"]
        intent = analysis["intent"]
        query_type = analysis["query_type"]
        
        # Add context based on intent and type
        enhancements = []
        
        if intent == 'definition':
            enhancements.append("Provide a clear definition and explanation")
        elif intent == 'tutorial':
            enhancements.append("Include step-by-step instructions")
        elif intent == 'comparison':
            enhancements.append("Compare and contrast the different options")
        elif intent == 'troubleshooting':
            enhancements.append("Focus on solutions and fixes")
        
        if query_type == 'example':
            enhancements.append("Include practical examples")
        elif query_type == 'list':
            enhancements.append("Provide a comprehensive list")
        
        # Role-based enhancements
        user_role = analysis["user_role"]
        if user_role == "Admin":
            enhancements.append("Include technical details and advanced information")
        elif user_role == "Guest":
            enhancements.append("Keep explanation simple and accessible")
        
        # Combine original query with enhancements
        if enhancements:
            enhanced_query = f"{query}. {'; '.join(enhancements)}."
        else:
            enhanced_query = query
        
        return enhanced_query

    def expand_query(self, query: str) -> List[str]:
        """
        Generate alternative query formulations for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded query variations
        """
        variations = [query]
        
        try:
            # Synonym expansion
            synonyms = {
                'machine learning': ['ML', 'artificial intelligence', 'AI', 'automated learning'],
                'artificial intelligence': ['AI', 'machine learning', 'ML', 'intelligent systems'],
                'deep learning': ['neural networks', 'DL', 'deep neural networks'],
                'algorithm': ['method', 'procedure', 'technique', 'approach'],
                'model': ['algorithm', 'system', 'framework', 'approach']
            }
            
            query_lower = query.lower()
            for term, alternatives in synonyms.items():
                if term in query_lower:
                    for alt in alternatives:
                        variation = query_lower.replace(term, alt)
                        variations.append(variation)
            
            # Remove duplicates
            variations = list(set(variations))
            
            return variations[:5]  # Limit to 5 variations
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]

    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract filtering criteria from the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary of filter criteria
        """
        filters = {}
        
        try:
            # Date filters
            date_patterns = [
                (r'after (\d{4})', 'after_year'),
                (r'before (\d{4})', 'before_year'),
                (r'in (\d{4})', 'year'),
                (r'recent', 'recent'),
                (r'latest', 'latest')
            ]
            
            for pattern, filter_type in date_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    if filter_type in ['after_year', 'before_year', 'year']:
                        filters[filter_type] = int(match.group(1))
                    else:
                        filters[filter_type] = True
            
            # Document type filters
            type_patterns = [
                (r'\bpdf\b', 'pdf'),
                (r'\bdocument\b', 'document'),
                (r'\barticle\b', 'article'),
                (r'\bpaper\b', 'paper'),
                (r'\bbook\b', 'book')
            ]
            
            for pattern, doc_type in type_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    filters['document_type'] = doc_type
                    break
            
            # Source filters
            if 'financial' in query.lower():
                filters['requires_permission'] = 'financial_data'
            
            return filters
            
        except Exception as e:
            logger.error(f"Error extracting filters: {e}")
            return {}

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate query for safety and appropriateness.
        
        Args:
            query: User query to validate
            
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'suggestions': []
        }
        
        try:
            # Length validation
            max_length = self.config["security"]["max_query_length"]
            if len(query) > max_length:
                validation['is_valid'] = False
                validation['issues'].append(f"Query too long (max {max_length} characters)")
            
            # Content validation
            if not query.strip():
                validation['is_valid'] = False
                validation['issues'].append("Empty query")
            
            # Check for potentially harmful content
            harmful_patterns = [
                r'<script', r'javascript:', r'eval\(', r'exec\(',
                r'DROP TABLE', r'DELETE FROM', r'UPDATE.*SET'
            ]
            
            for pattern in harmful_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    validation['is_valid'] = False
                    validation['issues'].append("Potentially harmful content detected")
                    break
            
            # Suggestions for improvement
            if len(query.split()) < 3:
                validation['suggestions'].append("Consider providing more details for better results")
            
            if not query.endswith('?') and not any(word in query.lower() for word in ['show', 'list', 'explain']):
                validation['suggestions'].append("Consider phrasing as a question for clearer intent")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return {'is_valid': False, 'issues': [f"Validation error: {e}"], 'suggestions': []}

    def get_query_context(self, query: str, conversation_history: List[Dict]) -> str:
        """
        Generate contextual information for the query based on conversation history.
        
        Args:
            query: Current query
            conversation_history: Previous conversation exchanges
            
        Returns:
            Contextual information string
        """
        try:
            if not conversation_history:
                return ""
            
            # Look for related topics in recent conversation
            recent_topics = []
            for exchange in conversation_history[-3:]:  # Last 3 exchanges
                if 'query' in exchange:
                    entities = self._extract_entities(exchange['query'])
                    recent_topics.extend(entities)
            
            # Find common topics with current query
            current_entities = self._extract_entities(query)
            common_topics = set(recent_topics) & set(current_entities)
            
            if common_topics:
                context = f"Related to previous discussion about: {', '.join(common_topics)}"
                return context
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting query context: {e}")
            return ""
