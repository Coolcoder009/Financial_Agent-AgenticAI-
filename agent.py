from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions= ["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

finance_agent=Agent(
    name="Financial Data Search Agent",
    role="Search the data for Finance stuffs",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True),],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Use the web search agent to find information about the company and the financial agent to find information about the stock.",
                  "Use table to display the data"],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Between NVIDIA and Tesla which stock to buy based on Financial analysis", stream=False)