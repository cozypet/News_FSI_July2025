"""
Financial Research Demo
Clean 5-block UI with manual search and dual suggestion inputs + archived article content display
"""

import streamlit as st
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

def get_secret(key):
    """Get secret from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for deployment)
        return st.secrets[key]
    except:
        # Fall back to environment variables (for local development)
        return os.getenv(key)

class FinancialResearchDemo:
    """Demo app with 5-block structure"""
    
    def __init__(self):
        """Initialize all clients"""
        self.mistral_client = MistralClient(api_key=get_secret("MISTRAL_API_KEY"))
        self.tavily_client = TavilyClient(api_key=get_secret("TAVILY_API_KEY"))
        
        # MongoDB connection
        self.mongo_client = MongoClient(get_secret("MONGO_CONNECTION_STRING"))
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
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return f"Analysis error occurred. Please check your Mistral AI connection and try again."

def format_live_results(results: List[Dict]) -> str:
    """Format live news results"""
    if not results:
        return "No live news found for this query."
    
    output = f"({len(results)} articles)\n\n"
    
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
    
    output = f"({len(results)} results) [clickable]\n\n"
    
    for i, result in enumerate(results, 1):
        metadata = result.get('metadata', {})
        title = metadata.get('title', 'No title')
        era = metadata.get('era', 'unknown').upper()
        year = metadata.get('published_date', '')[:4] if metadata.get('published_date') else 'Unknown'
        source = metadata.get('source', 'Unknown')
        score = result.get('score', 0)
        
        output += f"**{i}. {title}**\n"
        output += f"🏛️ *{era} ERA ({year})* • 📰 *{source}* • 📊 *{score:.3f}*\n\n"
    
    # Get content from the most relevant (first) article
    first_article = results[0]
    metadata = first_article.get('metadata', {})
    content = first_article.get('content', 'No content available.')
    title = metadata.get('title', 'No title')
    era = metadata.get('era', 'unknown').upper()
    year = metadata.get('published_date', '')[:4] if metadata.get('published_date') else 'Unknown'
    source = metadata.get('source', 'Unknown')
    
    content_display = f"**{title}**\n\n"
    content_display += f"🏛️ *{era} ERA ({year})* • 📰 *{source}*\n\n"
    content_display += f"---\n\n{content}"
    
    return output, content_display

def create_suggestion_cards(suggestions: List[str]) -> str:
    """Create suggestion cards HTML for Streamlit"""
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
            text-align: center;
        ">
            <div style="font-weight: 600; color: #1f4e79; margin-bottom: 8px;">💡 Suggestion {i+1}</div>
            <div style="color: #495057; font-size: 14px;">{suggestion}</div>
        </div>
        """
    
    cards_html += "</div>"
    return cards_html

def main_search(demo_app, query: str) -> Tuple[str, str, str, str, List[str]]:
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

def analyze_suggestion(demo_app, suggestion_idx: int, query: str, suggestions: List[str]) -> str:
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

def analyze_custom(demo_app, custom_query: str, original_query: str) -> str:
    """Analyze custom user input"""
    if not custom_query.strip():
        return "Please enter your analysis question."
    
    # Get fresh results for analysis
    live_results = demo_app.search_live_news(original_query)
    archived_results = demo_app.search_archived_news(original_query)
    
    # Generate deep analysis
    analysis = demo_app.deep_analysis(custom_query, original_query, live_results, archived_results)
    
    return f"## 🤖 Custom Analysis: {custom_query}\n\n{analysis}"

def main():
    """Create the enhanced 5-block Streamlit interface"""
    
    # Page configuration
    st.set_page_config(
        page_title="Global Bank Financial Research",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for styling
    css = """
    <style>
    .stApp {
        max-width: 1600px;
        margin: auto;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .main-header {
        text-align: center; 
        padding: 30px; 
        background: linear-gradient(135deg, #1f4e79, #2d5aa0); 
        color: white; 
        border-radius: 15px; 
        margin-bottom: 30px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79, #2d5aa0) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2d5aa0, #1f4e79) !important;
    }
    .suggestion-button {
        background: linear-gradient(90deg, #ff6b35, #f7931e) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    .suggestion-button:hover {
        background: linear-gradient(90deg, #f7931e, #ff6b35) !important;
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)
    
    # Initialize demo app
    if 'demo_app' not in st.session_state:
        st.session_state.demo_app = FinancialResearchDemo()
    
    # ┌─────────────────────────────────────┐
    # │ 🏛️ Global Bank AI Research          │
    # │ Transform 4 hours → 4 seconds       │
    # └─────────────────────────────────────┘
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5em; font-weight: 300;">🏛️  Global Bank Asset Management Intelligence Platform</h1>
        <p style="margin: 10px 0; font-size: 1.3em; color: #ffd700; font-weight: 500;">Transform 4 hours → 4 seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ┌─────────────────────────────────────┐
    # │ 🔍 Query Input + Search Button      │
    # └─────────────────────────────────────┘
    st.markdown("### 🔍 Investment Research Query")
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "",
            placeholder="Research query (e.g., 'Fed policy rate news')",
            key="search_query",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Search functionality
    if search_button and query:
        with st.spinner("🔍 Searching for financial insights..."):
            # Perform both live and archived searches
            live_results = st.session_state.demo_app.search_live_news(query)
            archived_results = st.session_state.demo_app.search_archived_news(query)
            
            # Format results
            live_output = format_live_results(live_results)
            archived_output, archived_content = format_archived_results_with_content(archived_results)
            
            # Generate suggestions
            suggestions = st.session_state.demo_app.generate_suggestions(query, live_results, archived_results)
            suggestions_html = create_suggestion_cards(suggestions)
            
            # Store results in session state
            st.session_state.live_results = live_results
            st.session_state.archived_results = archived_results
            st.session_state.live_output = live_output
            st.session_state.archived_output = archived_output
            st.session_state.archived_content = archived_content
            st.session_state.suggestions_html = suggestions_html
            st.session_state.suggestions = suggestions
            st.session_state.current_query = query
    
    st.markdown("---")
    
    # ┌─────────────┬─────────────┬─────────────┐
    # │ 🔴 Live     │ 📚 Archived │ 📄 Article  │
    # │ Market News │ Historical  │ Content     │
    # │             │ Analysis    │             │
    # │ (5 results) │ (5 results) │ (full text) │
    # │             │ [clickable] │ [selected]  │
    # └─────────────┴─────────────┴─────────────┘
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🔴 Real-Time Market Intelligence")
        if 'live_output' in st.session_state:
            st.markdown(st.session_state.live_output)
        else:
            st.info("Enter a query and click search to see live news results.")
    
    with col2:
        st.markdown("### 📚 Archived Historical Analysis")
        st.markdown("*(5 results) [clickable]*")
        if 'archived_results' in st.session_state and st.session_state.archived_results:
            for i, result in enumerate(st.session_state.archived_results, 1):
                metadata = result.get('metadata', {})
                title = metadata.get('title', 'No title')
                era = metadata.get('era', 'unknown').upper()
                year = metadata.get('published_date', '')[:4] if metadata.get('published_date') else 'Unknown'
                source = metadata.get('source', 'Unknown')
                score = result.get('score', 0)
                
                # Create clickable button for each archived article
                if st.button(f"{i}. {title[:50]}...", key=f"archived_{i}", use_container_width=True):
                    st.session_state.selected_article = result
                
                st.markdown(f"🏛️ *{era} ERA ({year})* • 📰 *{source}* • 📊 *{score:.3f}*")
                st.markdown("---")
        else:
            st.info("Enter a query and click search to see historical analysis.")
    
    with col3:
        st.markdown("### 📄 Article Content")
        if 'selected_article' in st.session_state:
            # Show selected article content
            article = st.session_state.selected_article
            metadata = article.get('metadata', {})
            content = article.get('content', 'No content available.')
            title = metadata.get('title', 'No title')
            era = metadata.get('era', 'unknown').upper()
            year = metadata.get('published_date', '')[:4] if metadata.get('published_date') else 'Unknown'
            source = metadata.get('source', 'Unknown')
            
            st.markdown(f"**{title}**")
            st.markdown(f"🏛️ *{era} ERA ({year})* • 📰 *{source}*")
            st.markdown("---")
            st.markdown(content)
        elif 'archived_content' in st.session_state:
            # Show default top match content
            st.markdown(st.session_state.archived_content)
        else:
            st.info("Click on an archived article to view its full content here.")
    
    st.markdown("---")
    
    # ┌─────────────────────────────────────┐
    # │ 💡 AI Analysis & Insights           │
    # │                                     │
    # │ 🤖 AI Suggested Questions:          │
    # │ [Card 1] [Card 2] [Card 3] [Card 4] │
    # │                                     │
    # │ ✏️ Ask Your Own Question:           │
    # │ ┌─────────────────────────────────┐ │
    # │ │ Type custom insight question... │ │
    # │ └─────────────────────────────────┘ │
    # │              [Analyze] →            │
    # │                                     │
    # │ 💡 Example: "Compare to 2008 crisis"│
    # └─────────────────────────────────────┘
    st.markdown("### 💡 AI Analysis & Insights")
    
    # AI Suggested Questions section
    st.markdown("#### 🤖 AI Suggested Questions:")
    
    if 'suggestions' in st.session_state and st.session_state.suggestions:
        # Display suggestion cards in a 2x2 grid
        col1, col2 = st.columns(2)
        for i, suggestion in enumerate(st.session_state.suggestions):
            with col1 if i % 2 == 0 else col2:
                if st.button(f"💡 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                    with st.spinner("🤖 Generating analysis..."):
                        analysis = analyze_suggestion(
                            st.session_state.demo_app,
                            i,
                            st.session_state.current_query,
                            st.session_state.suggestions
                        )
                        st.session_state.current_analysis = analysis
                        st.session_state.analysis_title = suggestion
    else:
        st.info("AI suggested questions will appear here after search...")
    
    st.markdown("")  # Add spacing
    
    # Ask Your Own Question section
    st.markdown("#### ✏️ Ask Your Own Investment Analysis Question:")
    col1, col2 = st.columns([4, 1])
    with col1:
        custom_query = st.text_input(
            "",
            placeholder="Type custom insight question...",
            key="custom_analysis",
            label_visibility="collapsed"
        )
    with col2:
        custom_btn = st.button("Analyze →", use_container_width=True)
    
    # Example suggestion
    st.markdown("💡 **Example:** *Compare to 2008 crisis*")
    
    if custom_btn and custom_query:
        with st.spinner("🤖 Generating custom analysis..."):
            analysis = analyze_custom(
                st.session_state.demo_app,
                custom_query,
                st.session_state.current_query
            )
            st.session_state.current_analysis = analysis
            st.session_state.analysis_title = custom_query
    
    st.markdown("---")
    
    # ┌─────────────────────────────────────┐
    # │ 🤖 Deep Analysis Results            │
    # │ Detailed Mistral insights appear    │
    # │ when cards clicked or custom        │
    # │ questions submitted                 │
    # └─────────────────────────────────────┘
    st.markdown("### 🤖 Deep Analysis Results")
    
    if 'current_analysis' in st.session_state:
        st.markdown(f"**Analysis: {st.session_state.analysis_title}**")
        # Use expandable container to ensure full text display
        with st.expander("📖 Full Analysis", expanded=True):
            st.markdown(st.session_state.current_analysis)
    else:
        st.info("Detailed Mistral insights appear when cards clicked or custom questions submitted")
    
    st.markdown("---")
    
    # ┌─────────────────────────────────────┐
    # │ 🚀 Business Impact                  │
    # │ 1,800x faster • $1.6M savings      │
    # │ 45+ years data • Unique insights   │
    # └─────────────────────────────────────┘
    st.markdown("""
    <div style="text-align: center; padding: 25px; background: #f8f9fa; border-radius: 10px; margin-top: 30px;">
        <h3 style="color: #1f4e79; margin-bottom: 15px;">🚀 Business Impact</h3>
        <p style="font-size: 1.2em; margin: 0; color: #495057; font-weight: 500;">
            <strong>1,800x faster decision speed </strong> • <strong>$1.6M analyst productivity gain</strong><br>
            <strong>45+ years market history</strong> • <strong>Proprietary cross-cycle analysis</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    print("🚀 Starting  Global Bank Financial Research Demo...")
    
    # Quick connection test would go here if needed
    print("🎬 Streamlit Demo launching...")
    print("🌐 Access with: streamlit run streamlit_demo_app.py")
    
    main()
