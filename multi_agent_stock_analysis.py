from agno.agent import Agent
from agno.team.team import Team
from agno.models.groq import Groq
from agno.tools import tool
from agno.tools.thinking import ThinkingTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.googlesearch import GoogleSearchTools
import agno

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
agno.app = os.getenv("AGNO_API_KEY")

# Define the sectors and their respective top metrics
sector_top_metrics = {
    "Banking & Financials": ["ROE", "Debt-to-Equity", "P/E Ratio", "EPS"],
    "IT Services/Tech Sector": [
        "P/E Ratio",
        "EPS",
        "ROE",
        "Net Profit Margin",
    ],
    "FMCG": ["Revenue Growth(YoY)", "Net Profit Margin", "P/E Ratio", "ROE"],
    "Automobiles & Auto Ancillaries": [
        "Revenue Growth(YoY)",
        "ROE",
        "P/E Ratio",
        "EPS",
    ],
    "Pharma & Healthcare": ["Revenue Growth(YoY)", "EPS", "P/E Ratio", "ROE"],
    "Infrastructure & Construction": [
        "Debt-to-Equity",
        "Revenue Growth(YoY)",
        "EPS",
        "ROE",
    ],
    "Oil, Gas & Energy": ["P/E Ratio", "EPS", "Dividend Yield", "ROE"],
    "Telecom": ["Debt-to-Equity", "Revenue Growth(YoY)", "P/E Ratio", "EPS"],
    "Metals & Mining": ["Revenue Growth(YoY)", "P/E Ratio", "EPS", "ROE"],
    "Real Estate": ["Debt-to-Equity", "Revenue Growth(YoY)", "P/E Ratio", "EPS"],
}


@tool(
    name="fetch_sector_wise_financial_metrics",
    description="Fetch the key financial metrics for a given stock based on its sector.",
)
def fetch_sector_wise_financial_metrics(stock_name):
    financial_metrics_agent = Agent(
        name="Key Financial Metrics Search Agent",
        role="Search the web for financial metrics related to stocks",
        description="Fetch latest financial metrics about company stocks using Google Search",
        model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(),
        ],
        instructions=[
            f"First, identify the financial metrics to fetch based on the sector of the stock from this \n{sector_top_metrics}\n"
            "Never use brave_search function to search anything.",
            "Then fetch the latest values of the identified financial metrics of the company stock using Google Search.",
            "Debt-to-Equity, Dividend Yield, EPS, P/E Ratio and ROE are available on Groww.in.",
            "Net Profit Margin is available on Moneycontrol.com.",
            "Revenue Growth(YoY) is available by just searching on Google.",
            "Use table format for the output.",
            "Always include sources.",
        ],
        expected_output="The latest values of the identified financial metrics for the given stock based on its sector.",
        show_tool_calls=True,
        markdown=True,
    )

    result = financial_metrics_agent.run(
        f"Fetch the key financial metrics for {stock_name} based on its sector."
    )
    return result.content


# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for news and information affecting stock prices",
    description="Fetch latest news and information about companies using Google Search",
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    tools=[
        ThinkingTools(add_instructions=True),
        GoogleSearchTools(),
    ],
    instructions=[
        "Never use brave_search function to search anything.",
        "Fetch latest company-related news using Google Search that affects their valuation.",
        "Always include sources.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Financial Data Agent
financial_data_agent = Agent(
    name="Financial Data Agent",
    role="Fetch and analyze financial data of individual stocks",
    description="Analyze individual stocks using financial data and metrics",
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[
        ThinkingTools(add_instructions=True),
        fetch_sector_wise_financial_metrics,
        YFinanceTools(
            stock_price=True,
            company_info=True,
            income_statements=True,
            analyst_recommendations=True,
            company_news=True,
            historical_prices=True,
        ),
    ],
    instructions=[
        "Focus on analyzing individual stocks.",
        "Never use brave_search function to search anything.",
        "Use the 'fetch_sector_wise_financial_metrics' tool to get the key financial metrics for the stocks in the portfolio and analyze based on those metrics.",
        f"Provide relevant metrics based on the sector as listed here \n{sector_top_metrics}\n",
        "Summarize the analysis with actionable insights.",
        "Always include sources.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Portfolio Analysis Agent
portfolio_analysis_agent = Agent(
    name="Portfolio Analysis Agent",
    role="Analyze and recommend stock portfolios",
    description="Analyze stock portfolios and provide recommendations based on financial metrics and diversification strategies",
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    tools=[
        ThinkingTools(add_instructions=True),
        fetch_sector_wise_financial_metrics,
        YFinanceTools(
            stock_price=False,
            company_info=True,
            income_statements=True,
            analyst_recommendations=True,
            historical_prices=True,
        ),
    ],
    instructions=[
        "Focus on portfolio-level analysis, including risk-return trade-offs.",
        "Never use brave_search function to search anything.",
        "Use the 'fetch_sector_wise_financial_metrics' tool to get the key financial metrics for the stocks in the portfolio and analyze based on those metrics.",
        f"Provide relevant metrics based on the sector as listed here \n{sector_top_metrics}\n",
        "Recommend portfolio adjustments based on financial goals.",
        "Always include sources.",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Agent to fetch key financial metrics
# financial_metrics_agent = Agent(
#     name="Key Financial Metrics Search Agent",
#     role="Search the web for financial metrics related to stocks",
#     description="Fetch latest financial metrics about company stocks using Google Search",
#     model=Groq(id="deepseek-r1-distill-llama-70b"),
#     tools=[
#         ReasoningTools( add_instructions=True),
#         GoogleSearchTools(),
#     ],
#     instructions=[
#         "Fetch latest financial metrics about company stocks using Google Search.",
#         "Use groww.in as the primary source for financial metrics.",
#         "Always include sources",
#     ],
#     show_tool_calls=True,
#     markdown=True,
# )

# Coordinator Agent
coordinator_agent = Team(
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    mode="coordinate",
    members=[
        web_search_agent,
        financial_data_agent,
        portfolio_analysis_agent,
    ],
    tools=[
        ThinkingTools(add_instructions=True),
    ],
    description="You are a senior financial analyst coordinating a team of agents to provide detailed stock market analysis and recommendations.",
    instructions=[
        "Analyze the prompt to determine if it is for a single stock, stock comparison, or portfolio analysis.",
        "Coordinate with the agents to fetch and analyze the required data.",
        f"Provide relevant metrics based on the sector as listed here \n{sector_top_metrics}\n",
        "Generate a detailed report with the following structure:",
        "1. Begin with the financial data table containing the relevant metrics.",
        "2. Next display the comparison in company insights/news affecting its valuation.",
        "3. Third section must focus on performance evaluation summary in a few words.",
        "4. Fourth section must have the analyst recommendations and your own insights.",
        "5. Finally, display the sources used to gather the data.",
        "Use tables to display data wherever possible.",
        "Always include sources from where the data was fetched.",
    ],
    show_tool_calls=True,
    markdown=True,
    expected_output="A detailed report as per the instructions.",
    show_members_responses=False,
)

if __name__ == "__main__":
    # User prompt
    try:
        coordinator_agent.print_response(
            "Compare TCS and Infosys and generate a comparison report."
        )
    except Exception as e:
        print(
            f"Sorry! An error occurred while processing your request. Please try again later. {e}"
        )
