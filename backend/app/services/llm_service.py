
import os
import logging
from typing import List, Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the official College AI Knowledge Assistant for our college.
Your mission is to provide accurate, helpful, and strictly verified answers to students, faculty, and visitors based ONLY on the official college documents provided in the context below.

CRITICAL GUIDELINES:
1. STRICT GROUNDING: You MUST base your answers ONLY on the provided context snippets.
2. NO HALLUCINATIONS / FABRICATION: If the question cannot be answered using the provided context, or if the context does not contain sufficient details, you MUST explicitly respond with:
   "I could not find information regarding this in the available college documents. Please consult the relevant department or administrative office for official confirmation."
   DO NOT make up fee amounts, exam dates, subject names, faculty details, placement packages, attendance rules, or policies.
3. CITATIONS & SOURCES: Always reference the specific document name and page number when stating facts (e.g., "[Source: AICTE-Mandatory-Disclosure.pdf, Page 4]").
4. FORMATTING: Use clean, student-friendly Markdown with bullet points, bold key terms, and tables when presenting schedules, syllabus structures, or fee breakdowns.
5. TONE: Professional, welcoming, encouraging, and clear.
6. COMPLETENESS: Finish every sentence and every bullet point. Never stop midway through a number, salary, date, or sentence.
7. EXACT NUMBERS: Preserve all figures exactly as written in the source documents. For example, 400000 must be written as ₹4,00,000.
8. COMPLETENESS CHECK: Before responding, verify that all requested figures and relevant details from the context are included.
"""

class LLMService:
    def __init__(self):
        raw_name = settings.LLM_MODEL
        self.model_name = raw_name.replace("models/", "")
        self._configured = False
        self._client = None
        self._check_configuration()

    def _check_configuration(self):
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if api_key and api_key.strip() and api_key != "your_google_gemini_api_key_here":
            try:
                from google import genai
                self._client = genai.Client(api_key=api_key.strip())
                self._configured = True
            except Exception as e:
                logger.error(f"Error configuring Google GenAI Client for LLM: {e}")
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=api_key.strip())
                    self._configured = True
                except Exception as e_legacy:
                    logger.error(f"Error configuring legacy genai: {e_legacy}")
                    self._configured = False
        else:
            self._configured = False

    def is_ready(self) -> bool:
        self._check_configuration()
        return self._configured

    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Constructs the augmented prompt combining history, retrieved context, and the user question."""

        # Build Context Section
        if context_chunks:
            context_blocks = []
            for i, chunk in enumerate(context_chunks, 1):
                doc_name = chunk.get("document_filename", "Document")
                category = chunk.get("category", "General")
                page = chunk.get("page_number", "N/A")
                text = chunk.get("text", "").strip()

                context_blocks.append(
                    f"--- CONTEXT SNIPPET #{i} ---\n"
                    f"Document: {doc_name}\n"
                    f"Category: {category}\n"
                    f"Page: {page}\n"
                    f"Content:\n{text}\n"
                )
            context_text = "\n".join(context_blocks)
        else:
            context_text = "No relevant context found in the college knowledge base."

        # Build Conversation History Section (last few turns)
        history_text = ""
        if conversation_history:
            recent_turns = conversation_history[-4:]  # Last 2 exchanges
            history_lines = []
            for turn in recent_turns:
                role = "Student" if turn.get("role") == "user" else "Assistant"
                content = turn.get("content", "")
                history_lines.append(f"{role}: {content}")
            if history_lines:
                history_text = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"""{history_text}OFFICIAL COLLEGE DOCUMENTS CONTEXT:
====================================
{context_text}
====================================

STUDENT QUESTION:
{query}

INSTRUCTIONS FOR YOUR RESPONSE:
1. Answer the student's question accurately using ONLY the official context above.
2. If the context does NOT contain the answer, politely state that the information was not found in the available college documents.
3. Cite the document names and page numbers in your answer wherever applicable.
4. Format your answer with clear Markdown headings and bullet points.
5. Finish every sentence and bullet point completely.
6. Preserve all numbers exactly as provided in the source documents.
7. Check that all requested figures and relevant details are included before responding.
"""
        return prompt

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generates an answer using Google Gemini."""
        if not self.is_ready():
            return (
                "⚠️ **Gemini API Key is not configured.**\n\n"
                "Please configure your `GEMINI_API_KEY` in the `backend/.env` file to enable AI answers."
            )

        prompt = self.build_prompt(query, context_chunks, conversation_history)
        import time

        last_error = None
        for attempt in range(3):
            try:
                if self._client:
                    from google.genai import types
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.2,
                            top_p=0.8,
                            max_output_tokens=2500,
                        )
                    )

                    if response:
                        logger.info(
                            "Gemini finish reason: %s",
                            getattr(response.candidates[0], "finish_reason", None)
                            if response.candidates else "No candidate"
                        )

                    if response and response.text:
                        return response.text.strip()

                # Legacy fallback
                import google.generativeai as legacy_genai
                model = legacy_genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT,
                    generation_config={
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "top_k": 40,
                        "max_output_tokens": 2500,
                    }
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    answer = response.text.strip()
                    logger.info("Generated answer length: %d", len(answer))
                    logger.info("Generated answer ending: %r", answer[-150:])
                    return answer
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API generation attempt {attempt + 1} failed: {e}")
                time.sleep(1.5)

        logger.error(f"Error generating answer with Gemini: {last_error}")
        err_str = str(last_error).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
            return "⚠️ **AI Rate Limit Reached**\n\nThe Google Gemini API daily/minute quota was temporarily exceeded. Please try again in a few moments."
        return f"An error occurred while generating the answer: {str(last_error)}"

llm_service = LLMService()
