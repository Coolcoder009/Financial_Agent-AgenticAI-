import os
import phi
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv


load_dotenv()
phi.api=os.getenv("PHI_KEY")

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

app=Playground(agents=[web_search_agent,finance_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)