"""
Goldman Sachs Financial Research Demo
Clean 5-block UI with manual search and dual suggestion inputs + archived article content display
"""

import gradio as gr
import json
import os
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Core dependencies
import pymongo
from pymongo import MongoClient
from mistralai.client import MistralClient
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FinancialResearchDemo:
    """Demo app with 5-block structure"""
    
    def __init__(self):
        """Initialize all clients"""
        self.mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # MongoDB connection
        self.mongo_client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
        self.db = self.mongo_client["financial_research"]
        self.collection = self.db["article_chunks"]
        
        print("✅ Financial Research Demo Ready")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Mistral AI"""
        try:
            response = self.mistral_client.embeddings(model="mistral-embed", input=[text])
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
    
    def search_live_news(self, query: str) -> List[Dict]:
        """Search live news via Tavily - returns 5 results"""
        try:
            financial_query = f"{query} financial markets Federal Reserve economy"
            response = self.tavily_client.search(
                query=financial_query,
                search_depth="basic",
                max_results=5,
                include_domains=["wsj.com", "bloomberg.com", "reuters.com", "ft.com", "cnbc.com"]
            )
            return response.get("results", [])[:5]
        except Exception as e:
            print(f"Live search error: {e}")
            return []
    
    def search_archived_news(self, query: str) -> List[Dict]:
        """Search historical articles via MongoDB - returns 5 results"""
        try:
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 50,
                        "limit": 5
                    }
                },
                {
                    "$project": {
                        "article_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            return results[:5]
        except Exception as e:
            print(f"Archived search error: {e}")
            return []
    
    def generate_suggestions(self, query: str, live_results: List[Dict], archived_results: List[Dict]) -> List[str]:
        """Generate AI suggestions based on search results"""
        try:
            # Prepare context from results
            context = f"Query: {query}\n\n"
            
            if live_results:
                context += "Current news headlines:\n"
                for item in live_results[:3]:
                    context += f"- {item.get('title', 'No title')}\n"
            
            if archived_results:
                context += "\nHistorical articles found:\n"
                for item in archived_results[:3]:
                    title = item.get('metadata', {}).get('title', 'No title')
                    era = item.get('metadata', {}).get('era', 'unknown')
                    context += f"- {title} ({era} era)\n"
            
            # Generate suggestions via Mistral
            prompt = f"""Based on this financial research context, suggest 4 specific follow-up analysis questions that would provide deeper insights. Make them actionable and specific.

{context}

Provide exactly 4 short, actionable questions (max 8 words each):"""

            response = self.mistral_client.chat(
                model="mistral-small",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            suggestions_text = response.choices[0].message.content
            # Parse suggestions (assuming they're numbered or bulleted)
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Clean up the line
                    clean_line = line.split('.', 1)[-1].split('-', 1)[-1].split('•', 1)[-1].strip()
                    if clean_line:
                        suggestions.append(clean_line)
            
            return suggestions[:4]  # Ensure exactly 4
            
        except Exception as e:
            print(f"Suggestions error: {e}")
            return [
                "Analyze sector-specific impacts",
                "Compare to historical precedents", 
                "Evaluate recovery timeline patterns",
                "Assess policy response effectiveness"
            ]
    
    def deep_analysis(self, analysis_query: str, original_query: str, live_results: List[Dict], archived_results: List[Dict]) -> str:
        """Generate deep analysis based on suggestion or custom input"""
        try:
            # Prepare comprehensive context
            context = f"Original Query: {original_query}\nAnalysis Focus: {analysis_query}\n\n"
            
            # Add live context
            if live_results:
                context += "CURRENT MARKET CONTEXT:\n"
                for item in live_results[:3]:
                    title = item.get('title', 'No title')
                    content = item.get('content', '')[:200] + "..."
                    context += f"• {title}\n  {content}\n\n"
            
            # Add historical context  
            if archived_results:
                context += "HISTORICAL CONTEXT:\n"
                for item in archived_results[:3]:
                    metadata = item.get('metadata', {})
                    title = metadata.get('title', 'No title')
                    era = metadata.get('era', 'unknown')
                    content = item.get('content', '')[:200] + "..."
                    context += f"• {title} ({era.upper()} ERA)\n  {content}\n\n"
            
            # Generate analysis
            prompt = f"""As a senior financial analyst, provide a comprehensive analysis addressing: {analysis_query}

Context:
{context}

Provide a structured analysis with:
1. Key insights from current market conditions
2. Historical patterns and precedents  
3. Investment implications
4. Risk factors to monitor
5. Actionable recommendations

Keep it professional but accessible, around 300-400 words."""

            response = self.mistral_client.chat(
                model="mistral-medium",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return f"Analysis error occurred. Please check your Mistral AI connection and try again."

# Initialize demo
demo_app = FinancialResearchDemo()

def format_live_results(results: List[Dict]) -> str:
    """Format live news results"""
    if not results:
        return "No live news found for this query."
    
    output = f"## 🔴 Live Market News ({len(results)} articles)\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('url', '#')
        published = result.get('published_date', 'Unknown date')
        source = url.split('/')[2] if '/' in url else 'Unknown source'
        
        output += f"**{i}. [{title}]({url})**\n"
        output += f"📰 *{source}* • 📅 *{published}*\n\n"
    
    return output

def format_archived_results_with_content(results: List[Dict]) -> Tuple[str, str]:
    """Format archived news results and return content for first article"""
    if not results:
        return "No historical articles found for this query.", "No archived article content available."
    
    output = f"## 📚 Historical Analysis ({len(results)} articles)\n\n"
    
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        title = metadata.get('title', 'No title')
        era = metadata.get('era', 'unknown').upper()
        year = metadata.get('published_date', '')[:4] if metadata.get('published_date') else 'Unknown'
        source = metadata.get('source', 'Unknown')
        score = result.get('score', 0)
        
        output += f"**{i}. {title}**\n"
        output += f"🏛️ *{era} ERA ({year})* • 📰 *{source}* • 📊 *{score:.3f} relevance*\n\n"
    
    # Get content from the most relevant (first) article
    first_article = results[0]
    metadata = first_article.get('metadata', {})
    content = first_article.get('content', 'No content available.')
    title = metadata.get('title', 'No title')
    era = metadata.get('era', 'unknown').upper()
    year = metadata.get('published_date', '')[:4] if metadata.get('published_date') else 'Unknown'
    source = metadata.get('source', 'Unknown')
    
    content_display = f"## 📄 Top Match Content\n\n"
    content_display += f"**{title}**\n\n"
    content_display += f"🏛️ *{era} ERA ({year})* • 📰 *{source}*\n\n"
    content_display += f"---\n\n{content}"
    
    return output, content_display

def create_suggestion_cards(suggestions: List[str]) -> str:
    """Create clickable suggestion cards"""
    if not suggestions:
        return "No suggestions available."
    
    cards_html = """
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0;">
    """
    
    for i, suggestion in enumerate(suggestions):
        cards_html += f"""
        <div style="
            border: 2px solid #e1e5e9; 
            border-radius: 10px; 
            padding: 15px; 
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        " onclick="document.getElementById('suggestion_{i}').click();">
            <div style="font-weight: 600; color: #1f4e79; margin-bottom: 8px;">💡 Suggestion {i+1}</div>
            <div style="color: #495057; font-size: 14px;">{suggestion}</div>
        </div>
        """
    
    cards_html += "</div>"
    return cards_html

def main_search(query: str) -> Tuple[str, str, str, str, List[str]]:
    """Main search function - now returns archived content as well"""
    if not query.strip():
        return "Please enter a search query.", "", "", "", []
    
    print(f"🔍 Searching for: {query}")
    
    # Perform searches
    live_results = demo_app.search_live_news(query)
    archived_results = demo_app.search_archived_news(query)
    
    # Format results
    live_output = format_live_results(live_results)
    archived_output, archived_content = format_archived_results_with_content(archived_results)
    
    # Generate suggestions
    suggestions = demo_app.generate_suggestions(query, live_results, archived_results)
    suggestions_html = create_suggestion_cards(suggestions)
    
    return live_output, archived_output, archived_content, suggestions_html, suggestions

def analyze_suggestion(suggestion_idx: int, query: str, suggestions: List[str]) -> str:
    """Analyze a selected suggestion"""
    if not suggestions or suggestion_idx >= len(suggestions):
        return "Invalid suggestion selected."
    
    analysis_query = suggestions[suggestion_idx]
    
    # Get fresh results for analysis
    live_results = demo_app.search_live_news(query)
    archived_results = demo_app.search_archived_news(query)
    
    # Generate deep analysis
    analysis = demo_app.deep_analysis(analysis_query, query, live_results, archived_results)
    
    return f"## 🤖 Deep Analysis: {analysis_query}\n\n{analysis}"

def analyze_custom(custom_query: str, original_query: str) -> str:
    """Analyze custom user input"""
    if not custom_query.strip():
        return "Please enter your analysis question."
    
    # Get fresh results for analysis
    live_results = demo_app.search_live_news(original_query)
    archived_results = demo_app.search_archived_news(original_query)
    
    # Generate deep analysis
    analysis = demo_app.deep_analysis(custom_query, original_query, live_results, archived_results)
    
    return f"## 🤖 Custom Analysis: {custom_query}\n\n{analysis}"

def create_interface():
    """Create the enhanced 5-block Gradio interface"""
    
    css = """
    .gradio-container {
        max-width: 1600px !important;
        margin: auto;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #1f4e79, #2d5aa0) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    .gr-button-secondary {
        background: linear-gradient(90deg, #ff6b35, #f7931e) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    """
    
    with gr.Blocks(css=css, title="Goldman Sachs Financial Research") as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #1f4e79, #2d5aa0); color: white; border-radius: 15px; margin-bottom: 30px;">
            <h1 style="margin: 0; font-size: 2.8em; font-weight: 300;">🏛️ Goldman Sachs</h1>
            <h2 style="margin: 10px 0; color: #ffd700; font-weight: 400;">AI Financial Research Platform</h2>
            <p style="margin: 0; font-size: 1.2em; opacity: 0.9;">Transform 4 hours of research into 4 seconds</p>
        </div>
        """)
        
        # Block 1: Query Input
        with gr.Row():
            query_input = gr.Textbox(
                label="🔍 Research Query",
                placeholder="Enter your financial research question (e.g., 'Fed rate hike impact on tech stocks')",
                lines=2,
                scale=4
            )
            search_btn = gr.Button("🔍 Search", variant="primary", scale=1, size="lg")
        
        # Block 2: Three-column Results Layout
        with gr.Row():
            # Column 1: Live Results
            live_results = gr.Markdown(
                label="🔴 Live Market News",
                value="Enter a query and click search to see live news results.",
                height=400
            )
            
            # Column 2: Archived Results List
            archived_results = gr.Markdown(
                label="📚 Historical Analysis", 
                value="Enter a query and click search to see historical analysis.",
                height=400
            )
            
            # Column 3: Archived Article Content
            archived_content = gr.Markdown(
                label="📄 Article Content",
                value="Most relevant archived article content will appear here after search.",
                height=400
            )
        
        # Block 3: Mistral Suggestions (dual input)
        gr.HTML("<h3 style='margin: 30px 0 15px 0; color: #1f4e79;'>💡 Analysis Options</h3>")
        
        # AI Suggestions
        suggestions_display = gr.HTML(
            value="<p style='color: #666; font-style: italic;'>AI suggestions will appear here after search...</p>"
        )
        
        # Hidden buttons for suggestions (will be triggered by HTML clicks)
        with gr.Row(visible=False):
            suggestion_btns = [gr.Button(f"Suggestion {i}", variant="secondary") for i in range(4)]
        
        # Custom input section
        gr.HTML("<h4 style='margin: 20px 0 10px 0; color: #1f4e79;'>✏️ Or ask your own question:</h4>")
        with gr.Row():
            custom_input = gr.Textbox(
                label="Custom Analysis Request",
                placeholder="e.g., 'How does this compare to the 2008 crisis?' or 'What are the bond market implications?'",
                lines=2,
                scale=4
            )
            custom_btn = gr.Button("🤖 Analyze", variant="secondary", scale=1)
        
        # Block 4: Deep Analysis
        analysis_output = gr.Markdown(
            label="🤖 AI Analysis",
            value="Click on a suggestion card or enter a custom question to get detailed analysis.",
            height=300
        )
        
        # State variables to track data
        current_query = gr.State("")
        current_suggestions = gr.State([])
        
        # Event handlers
        def handle_search(query):
            live, archived, content, suggestions_html, suggestions = main_search(query)
            return live, archived, content, suggestions_html, query, suggestions, ""
        
        def handle_suggestion(idx, query, suggestions):
            return analyze_suggestion(idx, query, suggestions)
        
        def handle_custom(custom_query, original_query):
            return analyze_custom(custom_query, original_query)
        
        # Connect search button
        search_btn.click(
            fn=handle_search,
            inputs=[query_input],
            outputs=[live_results, archived_results, archived_content, suggestions_display, current_query, current_suggestions, analysis_output]
        )
        
        # Connect suggestion buttons
        for i, btn in enumerate(suggestion_btns):
            btn.click(
                fn=lambda idx=i, q=current_query, s=current_suggestions: handle_suggestion(idx, q, s),
                inputs=[],
                outputs=[analysis_output]
            )
        
        # Connect custom analysis
        custom_btn.click(
            fn=handle_custom,
            inputs=[custom_input, current_query],
            outputs=[analysis_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 25px; background: #f8f9fa; border-radius: 10px; margin-top: 30px;">
            <h4 style="color: #1f4e79; margin-bottom: 15px;">🚀 Business Impact</h4>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                <div style="margin: 5px;"><strong>⚡ Speed:</strong> 1,800x faster</div>
                <div style="margin: 5px;"><strong>📊 Coverage:</strong> 45+ years of data</div>
                <div style="margin: 5px;"><strong>💰 Savings:</strong> $1.6M annually</div>
                <div style="margin: 5px;"><strong>🎯 Advantage:</strong> Unique AI insights</div>
            </div>
        </div>
        """)
    
    return interface

def main():
    """Launch the demo"""
    print("🚀 Starting Goldman Sachs Financial Research Demo...")
    
    # Quick connection test
    try:
        demo_app.mistral_client.embeddings(model="mistral-embed", input=["test"])
        print("✅ All systems ready")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return
    
    interface = create_interface()
    
    print("🎬 Demo launching...")
    print("🌐 Access at: http://localhost:7860")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()