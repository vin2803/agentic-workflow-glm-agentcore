from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_aws.llms import SagemakerEndpoint
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler
from bedrock_agentcore.runtime import BedrockAgentCoreApp
import argparse
import json
import re
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import boto3
import os
import tempfile

app = BedrockAgentCoreApp()

# Initialize S3 client
s3_client = boto3.client('s3')
S3_BUCKET_NAME = "gpt-oss-agentic-demo"

# Create stock analysis tools
@tool
def gather_stock_data(stock_symbol: str) -> str:
    """
    Gather comprehensive stock data from various sources including price history, 
    financial metrics, news, and market data.
    
    Args:
        stock_symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
    
    Returns:
        Comprehensive stock data including current price, historical performance, 
        financial metrics, and recent news
    """
    try:
        # Clean the stock symbol
        symbol = stock_symbol.upper().strip()
        
        # Get stock data using yfinance
        stock = yf.Ticker(symbol)
        
        # Get basic info
        info = stock.info
        
        # Get historical data (1 year)
        hist = stock.history(period="1y")
        current_price = hist['Close'].iloc[-1] if not hist.empty else 0
        
        # Calculate performance metrics
        if len(hist) > 0:
            year_high = hist['High'].max()
            year_low = hist['Low'].min()
            year_start_price = hist['Close'].iloc[0]
            ytd_return = ((current_price - year_start_price) / year_start_price) * 100
            
            # Calculate volatility (standard deviation of daily returns)
            daily_returns = hist['Close'].pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
        else:
            year_high = year_low = ytd_return = volatility = 0
            
        # Get recent news (simulated - in production you'd use a real news API)
        recent_news = [
            f"{symbol} reports quarterly earnings with mixed results",
            f"Analysts upgrade {symbol} price target amid strong fundamentals",
            f"{symbol} announces new strategic partnership",
            f"Market volatility affects {symbol} trading volume"
        ]
        
        # Format the comprehensive data
        stock_data = f"""STOCK DATA GATHERING REPORT:
================================
Stock Symbol: {symbol}
Company Name: {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}

CURRENT MARKET DATA:
- Current Price: ${current_price:.2f}
- Market Cap: ${info.get('marketCap', 0):,} 
- 52-Week High: ${year_high:.2f}
- 52-Week Low: ${year_low:.2f}
- YTD Return: {ytd_return:.2f}%
- Volatility (Annualized): {volatility:.2f}%

FINANCIAL METRICS:
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Forward P/E: {info.get('forwardPE', 'N/A')}
- Price-to-Book: {info.get('priceToBook', 'N/A')}
- Dividend Yield: {info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0:.2f}%
- Revenue (TTM): ${info.get('totalRevenue', 0):,}
- Profit Margin: {info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0:.2f}%

TRADING METRICS:
- Average Volume: {info.get('averageVolume', 0):,}
- Beta: {info.get('beta', 'N/A')}
- EPS (TTM): ${info.get('trailingEps', 'N/A')}
- Book Value: ${info.get('bookValue', 'N/A')}

RECENT NEWS HEADLINES:
{chr(10).join(f"- {news}" for news in recent_news)}

DATA COLLECTION TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return stock_data
        
    except Exception as e:
        return f"""STOCK DATA GATHERING ERROR:
================================
Stock Symbol: {stock_symbol}
Error: Unable to gather comprehensive stock data
Details: {str(e)}

Please verify the stock symbol is correct and try again.
"""

@tool
def analyze_stock_performance(stock_data: str) -> str:
    """
    Analyze stock performance based on gathered data, providing technical analysis,
    fundamental analysis, and risk assessment WITHOUT investment recommendations.
    
    Args:
        stock_data: Raw stock data from the data gathering agent
    
    Returns:
        Comprehensive stock analysis including technical indicators, fundamental analysis,
        and risk assessment for informational purposes only
    """
    import re
    
    # Extract key metrics from stock data
    symbol_match = re.search(r'Stock Symbol: ([A-Z]+)', stock_data)
    price_match = re.search(r'Current Price: \$([\d.]+)', stock_data)
    pe_match = re.search(r'P/E Ratio: ([\d.]+)', stock_data)
    ytd_match = re.search(r'YTD Return: ([\d.-]+)%', stock_data)
    volatility_match = re.search(r'Volatility \(Annualized\): ([\d.]+)%', stock_data)
    dividend_match = re.search(r'Dividend Yield: ([\d.]+)%', stock_data)
    beta_match = re.search(r'Beta: ([\d.]+)', stock_data)
    profit_margin_match = re.search(r'Profit Margin: ([\d.]+)%', stock_data)
    
    symbol = symbol_match.group(1) if symbol_match else 'UNKNOWN'
    current_price = float(price_match.group(1)) if price_match else 0
    pe_ratio = float(pe_match.group(1)) if pe_match and pe_match.group(1) != 'N/A' else None
    ytd_return = float(ytd_match.group(1)) if ytd_match else 0
    volatility = float(volatility_match.group(1)) if volatility_match else 0
    dividend_yield = float(dividend_match.group(1)) if dividend_match else 0
    beta = float(beta_match.group(1)) if beta_match and beta_match.group(1) != 'N/A' else None
    profit_margin = float(profit_margin_match.group(1)) if profit_margin_match else 0
    
    # Technical Analysis (descriptive only)
    if ytd_return > 20:
        price_trend = "STRONG UPTREND"
    elif ytd_return > 10:
        price_trend = "MODERATE UPTREND"
    elif ytd_return > 0:
        price_trend = "SLIGHT UPTREND"
    elif ytd_return > -10:
        price_trend = "SLIGHT DOWNTREND"
    else:
        price_trend = "STRONG DOWNTREND"
    
    # Fundamental Analysis (descriptive only)
    fundamental_factors = []
    
    if pe_ratio:
        if pe_ratio < 15:
            fundamental_factors.append("P/E ratio suggests potential undervaluation")
        elif pe_ratio < 25:
            fundamental_factors.append("P/E ratio within reasonable range")
        else:
            fundamental_factors.append("P/E ratio suggests potential overvaluation")
    
    if profit_margin > 20:
        fundamental_factors.append("Excellent profit margins")
    elif profit_margin > 10:
        fundamental_factors.append("Good profit margins")
    else:
        fundamental_factors.append("Low profit margins")
    
    if dividend_yield > 3:
        fundamental_factors.append("High dividend yield")
    elif dividend_yield > 1:
        fundamental_factors.append("Moderate dividend yield")
    else:
        fundamental_factors.append("Low or no dividend yield")
    
    
    beta_description = ""
    if beta and beta > 1.5:
        beta_description = "High beta indicates sensitivity to market movements"
    elif beta and beta < 0.5:
        beta_description = "Low beta indicates stability relative to market"
    else:
        beta_description = "Beta indicates moderate market correlation"
    
    analysis_report = f"""STOCK PERFORMANCE ANALYSIS:
===============================
Stock: {symbol} | Current Price: ${current_price:.2f}

TECHNICAL ANALYSIS:
- Price Trend: {price_trend}
- YTD Performance: {ytd_return:.2f}%


FUNDAMENTAL ANALYSIS:
- P/E Ratio: {pe_ratio if pe_ratio else 'N/A'}
- Profit Margin: {profit_margin:.2f}%
- Dividend Yield: {dividend_yield:.2f}%
- Beta: {beta if beta else 'N/A'}

KEY OBSERVATIONS:
{chr(10).join(f"• {factor}" for factor in fundamental_factors)}



ANALYST SUMMARY:
Based on technical and fundamental analysis, {symbol} shows {price_trend.lower()} with {risk_level.lower()} volatility profile. 
The analysis reflects current market conditions and financial performance metrics for informational purposes.

DISCLAIMER: This analysis is for informational purposes only and does not constitute investment advice.
"""
    
    return analysis_report

@tool
def generate_stock_report(stock_data: str, analysis_data: str) -> str:
    """
    Generate a comprehensive stock report based on gathered data and analysis.
    Creates a professional PDF report and uploads to S3 for documentation purposes.
    
    Args:
        stock_data: Raw stock data from the data gathering agent
        analysis_data: Analysis results from the performance analyzer
    
    Returns:
        Report generation summary with PDF creation and S3 upload status
    """
    import re
    
    # Extract key information for report
    symbol_match = re.search(r'Stock Symbol: ([A-Z]+)', stock_data)
    price_match = re.search(r'Current Price: \$([\d.]+)', stock_data)
    company_match = re.search(r'Company Name: ([^\n]+)', stock_data)
    sector_match = re.search(r'Sector: ([^\n]+)', stock_data)
    ytd_match = re.search(r'YTD Performance: ([\d.-]+)%', analysis_data)
    risk_match = re.search(r'Volatility Risk: ([A-Z]+)', analysis_data)
    
    symbol = symbol_match.group(1) if symbol_match else 'UNKNOWN'
    current_price = float(price_match.group(1)) if price_match else 0
    company_name = company_match.group(1).strip() if company_match else 'N/A'
    sector = sector_match.group(1).strip() if sector_match else 'N/A'
    ytd_performance = float(ytd_match.group(1)) if ytd_match else 0
    risk_level = risk_match.group(1) if risk_match else 'MEDIUM'
    
    # Generate PDF report and upload to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{symbol}_Stock_Report_{timestamp}.pdf"
    
    try:
        s3_path = create_and_upload_stock_report_pdf(
            symbol, company_name, sector, current_price, 
            ytd_performance, risk_level, stock_data, analysis_data, pdf_filename
        )
        pdf_status = f"PDF report uploaded to S3: {s3_path}"
    except Exception as e:
        pdf_status = f"PDF generation/upload failed: {str(e)}"
    
    report_summary = f"""STOCK REPORT GENERATION:
===============================
Stock: {symbol} ({company_name})
Sector: {sector}
Current Price: ${current_price:.2f}

REPORT SUMMARY:
- Technical Analysis: {ytd_performance:.2f}% YTD performance
- Risk Assessment: {risk_level} volatility risk
- Report Type: Comprehensive stock analysis for informational purposes
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{pdf_status}

REPORT CONTENTS:
• Executive Summary with key metrics
• Detailed market data and financial metrics
• Technical and fundamental analysis
• Risk assessment and observations
• Professional formatting for documentation

DISCLAIMER: This report is for informational and educational purposes only. 
It does not constitute investment advice or recommendations.
"""
    
    return report_summary

def create_and_upload_stock_report_pdf(symbol, company_name, sector, price, ytd_perf, risk_level, stock_data, analysis_data, filename):
    """Create a professional PDF stock report and upload to S3"""
    
    # Create PDF in temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        doc = SimpleDocTemplate(tmp_file.name, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph(f"Stock Analysis Report: {symbol}", title_style))
        story.append(Spacer(1, 12))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_data = [
            ['Metric', 'Value'],
            ['Stock Symbol', symbol],
            ['Company Name', company_name],
            ['Sector', sector],
            ['Current Price', f"${price:.2f}"],
            ['YTD Performance', f"{ytd_perf:.2f}%"],
            ['Risk Level', risk_level]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Stock Data Section
        story.append(Paragraph("Market Data", styles['Heading2']))
        story.append(Paragraph(stock_data.replace('\n', '<br/>'), styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Analysis Section
        story.append(Paragraph("Performance Analysis", styles['Heading2']))
        story.append(Paragraph(analysis_data.replace('\n', '<br/>'), styles['Normal']))
        
        # Generate timestamp
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph("This report is for informational purposes only.", styles['Normal']))
        
        doc.build(story)
        
        # Upload to S3
        s3_key = datetime.now().strftime('%Y/%m/%d') + "/" + filename
        
        try:
            s3_client.upload_file(tmp_file.name, S3_BUCKET_NAME, s3_key)
            s3_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return s3_path
            
        except Exception as e:
            # Clean up temporary file on error
            os.unlink(tmp_file.name)
            raise e

# Custom wrapper to make SagemakerEndpoint work with LangGraph tool binding
class SagemakerLLMWrapper:
    def __init__(self, sagemaker_llm, tools):
        self.sagemaker_llm = sagemaker_llm
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
    
    def bind_tools(self, tools):
        # Return self since we're already configured with tools
        return self
    
    def invoke(self, messages):
        # Extract the user message content
        user_content = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_content = msg.content
                break
        
        # Check if this is a stock analysis request
        if any(keyword in user_content.lower() for keyword in ['analyze', 'stock', 'ticker', 'symbol']):
            # Extract stock symbol from user input
            stock_match = re.search(r'\b([A-Z]{2,5})\b', user_content.upper())
            if stock_match:
                stock_symbol = stock_match.group(1)
                
                # Step 1: Gather stock data
                print(f"Step 1: Gathering data for {stock_symbol}...")
                stock_data = self.tools[0].invoke({"stock_symbol": stock_symbol})
                
                # Step 2: Analyze stock performance
                print(f"Step 2: Analyzing {stock_symbol} performance...")
                analysis_result = self.tools[1].invoke({"stock_data": stock_data})
                
                # Step 3: Generate stock report
                print(f"Step 3: Generating report for {stock_symbol}...")
                report_result = self.tools[2].invoke({"stock_data": stock_data, "analysis_data": analysis_result})
                
                # Return comprehensive response
                full_response = f"""**COMPREHENSIVE STOCK ANALYSIS REPORT**

**Step 1 - Stock Data Gathering:**
{stock_data}

**Step 2 - Performance Analysis:**
{analysis_result}

**Step 3 - Report Generation:**
{report_result}

---
**ANALYSIS COMPLETE:** Comprehensive stock analysis has been performed and a detailed PDF report has been generated and uploaded to S3 for documentation purposes."""
                
                return AIMessage(content=full_response)
            else:
                return AIMessage(content="Please provide a valid stock symbol (e.g., AAPL, GOOGL, TSLA) for analysis.")
        
        # For other messages, use the SageMaker model normally
        system_msg = """You are a professional stock analyst. Provide helpful responses about stock analysis, market trends, and financial metrics for informational purposes only."""
        
        full_prompt = f"{system_msg}\n\nUser: {user_content}"
        
        # Get response from SageMaker endpoint
        response = self.sagemaker_llm.invoke(full_prompt)
        
        # Return a proper LangChain AIMessage
        return AIMessage(content=response)

# Define the agent using SageMaker endpoint
def create_agent():
    """Create and configure the LangGraph stock analysis agent with SageMaker endpoint"""
    
    # Your SageMaker endpoint configuration
    endpoint_name = "gpt-oss-120b-2025-11-05-01-53-27-686"
    
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
            # GPT-OSS harmony format payload structure
            payload = {
                "model": "/opt/ml/model",
                "input": [
                    {
                        "role": "system",
                        "content": "You are a professional stock analyst. Analyze stocks and provide detailed information for educational purposes only, without investment recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_output_tokens": model_kwargs.get("max_new_tokens", 2048),
                "stream": "false",
                "temperature": model_kwargs.get("temperature", 0.1),
                "top_p": model_kwargs.get("top_p", 1)
            }
            input_str = json.dumps(payload)
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            # Parse harmony format response
            decoded_output = output.read().decode("utf-8")
            response_json = json.loads(decoded_output)
            
            if 'output' in response_json and isinstance(response_json['output'], list):
                for item in response_json['output']:
                    if item.get('type') == 'message' and item.get('role') == 'assistant':
                        content = item.get('content', [])
                        for content_item in content:
                            if content_item.get('type') == 'output_text':
                                return content_item.get('text', '')
                
                # Fallback parsing for different harmony format structures
                for item in response_json['output']:
                    if item.get('type') != 'reasoning' and 'content' in item and isinstance(item['content'], list):
                        for content_item in item['content']:
                            if content_item.get('type') == 'output_text' and 'text' in content_item:
                                return content_item['text']
                
                for item in response_json['output']:
                    if 'content' in item and isinstance(item['content'], list):
                        for content_item in item['content']:
                            if 'text' in content_item:
                                return content_item['text']
            
            return str(response_json)

    # Initialize SageMaker LLM with harmony format
    content_handler = ContentHandler()
    sagemaker_llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name="us-west-2",
        model_kwargs={
            "max_new_tokens": 2048, 
            "do_sample": True, 
            "temperature": 0.1,  # Lower temperature for consistent analysis
            "top_p": 1
        },
        content_handler=content_handler
    )
    
    # Create tools (3 tools: data gathering, analysis, report generation)
    tools = [gather_stock_data, analyze_stock_performance, generate_stock_report]
    
    # Wrap SageMaker LLM to work with LangGraph
    llm_with_tools = SagemakerLLMWrapper(sagemaker_llm, tools)
    
    # System message for stock analysis
    system_message = """You are a professional stock analyst with expertise in technical analysis, fundamental analysis, and report generation. 

Your role is to:
1. Gather comprehensive stock data from multiple sources including price history, financial metrics, and market data
2. Analyze stock performance using both technical and fundamental analysis techniques
3. Generate professional stock reports for documentation and educational purposes

Provide informational analysis only, without investment recommendations or advice."""
    
    # Define the chatbot node
    def chatbot(state: MessagesState):
        # Add system message if not already present
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_message)] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Create the graph
    graph_builder = StateGraph(MessagesState)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    
    # Set entry point
    graph_builder.set_entry_point("chatbot")
    
    # Compile the graph
    return graph_builder.compile()

# Initialize the agent
agent = create_agent()

@app.entrypoint
def langgraph_stock_sagemaker(payload):
    """
    Invoke the stock analysis agent with a payload
    """
    user_input = payload.get("prompt")
    
    # Create the input in the format expected by LangGraph
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    
    # Extract the final message content
    return response["messages"][-1].content

if __name__ == "__main__":
    app.run()
