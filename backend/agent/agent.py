"""Main agent implementation for ETF research"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from anthropic import Anthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.agent.tools import get_research_tools
from backend.agent.prompts import SYSTEM_PROMPT, ANSWER_SYNTHESIS_PROMPT
from backend.utils.config import get_config
from backend.utils.logger import setup_logger, StructuredLogger
from backend.utils.cost_monitor import get_cost_monitor
from backend.database.connection import get_db_manager
from backend.database.repository import QueryHistoryRepository, AgentEvaluationRepository

logger = StructuredLogger("agent")
cost_monitor = get_cost_monitor()


class ETFResearchAgent:
    """Agentic RAG system for ETF stock research"""

    def __init__(self):
        self.config = get_config().agent
        self.rag_config = get_config().rag
        self.tools_instance = get_research_tools()
        self.db_manager = get_db_manager()

        # Initialize LLM
        self._initialize_llm()

        # Create tools
        self._create_tools()

        # Create agent
        self._create_agent()

    def _initialize_llm(self):
        """Initialize the language model"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.llm = ChatAnthropic(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            anthropic_api_key=api_key,
        )

        logger.log_event(
            "agent_init",
            f"Initialized LLM: {self.config.model_name}",
            model=self.config.model_name,
        )

    def _create_tools(self):
        """Create LangChain tools from research tools"""
        self.tools = [
            Tool(
                name="search_documents",
                description=(
                    "Search for relevant documents using semantic search. "
                    "Use this for general queries about topics, trends, companies, or financial events. "
                    "Args: query (str), top_k (int, optional), ticker_filter (str, optional)"
                ),
                func=lambda query, **kwargs: json.dumps(
                    self.tools_instance.search_documents(query, **kwargs),
                    default=str
                ),
            ),
            Tool(
                name="get_stock_documents",
                description=(
                    "Get recent documents for a specific stock ticker. "
                    "Use when asked about a specific company. "
                    "Args: ticker (str), document_type (str, optional: 'news', 'earnings', 'company_info'), limit (int, optional)"
                ),
                func=lambda ticker, **kwargs: json.dumps(
                    self.tools_instance.get_stock_documents(ticker, **kwargs),
                    default=str
                ),
            ),
            Tool(
                name="get_ai_stocks_list",
                description=(
                    "Get list of all AI-related stocks in the database. "
                    "Use when asked about which AI stocks are available or for stock recommendations."
                ),
                func=lambda: json.dumps(
                    self.tools_instance.get_ai_stocks_list()
                ),
            ),
            Tool(
                name="get_etf_holdings",
                description=(
                    "Get holdings for a specific ETF. "
                    "Args: etf (str), ai_only (bool, optional)"
                ),
                func=lambda etf, **kwargs: json.dumps(
                    self.tools_instance.get_etf_holdings(etf, **kwargs),
                    default=str
                ),
            ),
            Tool(
                name="compare_stocks",
                description=(
                    "Compare multiple stocks based on available data. "
                    "Use when asked to compare companies. "
                    "Args: tickers (list of str), aspect (str, optional: 'general', 'news', 'earnings')"
                ),
                func=lambda tickers, **kwargs: json.dumps(
                    self.tools_instance.compare_stocks(tickers, **kwargs),
                    default=str
                ),
            ),
        ]

    def _create_agent(self):
        """Create the LangChain agent"""
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        logger.log_event("agent_created", "Agent executor created successfully")

    def query(
        self,
        question: str,
        return_metadata: bool = False,
    ) -> Dict[str, Any]:
        """Query the agent

        Args:
            question: User question
            return_metadata: Whether to return metadata (timing, costs, etc.)

        Returns:
            Response dictionary with answer and optional metadata
        """
        start_time = datetime.now()

        logger.log_event(
            "agent_query_start",
            f"Processing query: {question}",
            question=question,
        )

        # Check budget
        if not cost_monitor.check_budget("agent"):
            logger.logger.warning("Agent budget exceeded")
            return {
                "answer": "Sorry, the daily budget for agent queries has been exceeded. Please try again tomorrow.",
                "error": "budget_exceeded",
            }

        try:
            # Run agent
            result = self.agent_executor.invoke({"input": question})

            # Calculate timing
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Estimate and log cost (rough estimate based on tokens)
            # Note: In production, use actual token counts from API response
            input_tokens = len(question.split()) * 1.3  # Rough estimate
            output_tokens = len(result["output"].split()) * 1.3
            cost = cost_monitor.log_agent_cost(
                provider=self.config.provider,
                model=self.config.model_name,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
            )

            # Log query to database
            with self.db_manager.get_session() as session:
                query_history = QueryHistoryRepository.create_query(
                    session=session,
                    query=question,
                    results_count=0,  # Agent doesn't return count directly
                    top_tickers=[],
                    agent_used=True,
                    response_time_ms=duration_ms,
                    cost_usd=cost,
                )

            logger.log_event(
                "agent_query_complete",
                "Query completed successfully",
                question=question,
                duration_ms=duration_ms,
                cost_usd=cost,
            )

            response = {
                "answer": result["output"],
                "question": question,
            }

            if return_metadata:
                response["metadata"] = {
                    "duration_ms": duration_ms,
                    "cost_usd": cost,
                    "model": self.config.model_name,
                    "timestamp": datetime.now().isoformat(),
                }

            return response

        except Exception as e:
            logger.logger.error(f"Agent query failed: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "error": str(e),
            }

    def simple_rag_query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform simple RAG without agent (faster, cheaper)

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            Response dictionary
        """
        start_time = datetime.now()

        logger.log_event("rag_query_start", f"Processing RAG query: {question}")

        try:
            # Search for relevant documents
            if top_k is None:
                top_k = self.rag_config.top_k

            results = self.tools_instance.search_documents(
                query=question,
                top_k=top_k,
            )

            if not results:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "documents": [],
                }

            # Synthesize answer using LLM
            documents_text = "\n\n".join([
                f"Document {i+1} (Ticker: {doc['ticker']}, Similarity: {doc['similarity']:.2f}):\n{doc['content']}"
                for i, doc in enumerate(results)
            ])

            synthesis_prompt = ANSWER_SYNTHESIS_PROMPT.format(
                question=question,
                documents=documents_text,
            )

            response = self.llm.invoke(synthesis_prompt)
            answer = response.content

            # Calculate timing and cost
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            input_tokens = len(synthesis_prompt.split()) * 1.3
            output_tokens = len(answer.split()) * 1.3
            cost = cost_monitor.log_agent_cost(
                provider=self.config.provider,
                model=self.config.model_name,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
            )

            # Log to database
            with self.db_manager.get_session() as session:
                top_tickers = list(set([r["ticker"] for r in results if r["ticker"]]))
                QueryHistoryRepository.create_query(
                    session=session,
                    query=question,
                    results_count=len(results),
                    top_tickers=top_tickers,
                    agent_used=False,
                    response_time_ms=duration_ms,
                    cost_usd=cost,
                )

            logger.log_event(
                "rag_query_complete",
                "RAG query completed",
                duration_ms=duration_ms,
                results_count=len(results),
            )

            return {
                "answer": answer,
                "documents": results,
                "metadata": {
                    "duration_ms": duration_ms,
                    "cost_usd": cost,
                    "results_count": len(results),
                },
            }

        except Exception as e:
            logger.logger.error(f"RAG query failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "error": str(e),
            }


def get_agent() -> ETFResearchAgent:
    """Get ETF research agent instance"""
    return ETFResearchAgent()
