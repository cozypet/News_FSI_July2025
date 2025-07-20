"""
Unified Document Processing Pipeline using Mistral Document AI
Handles both JSON articles and PDF newspaper files with automatic detection

This module provides a unified interface for processing:
- JSON Articles → Direct processing → Existing Pipeline
- PDF Newspapers → Mistral Document AI → Text Extraction → Existing Pipeline
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import existing processor
from data_prep import ArticleProcessor, ArticleChunk

# Core dependencies
from mistralai.client import MistralClient
import pymongo
from pymongo import MongoClient

@dataclass
class PDFArticleExtraction:
    """Data class for extracted articles from PDF"""
    article_id: str
    title: str
    content: str
    section: str
    page_number: int
    column_position: str
    extracted_metadata: Dict[str, Any]
    confidence_score: float
    extraction_timestamp: str

class UnifiedDocumentProcessor:
    """
    Unified processor that handles both JSON articles and PDF newspapers
    
    Auto-detects file type and routes to appropriate processing:
    
    JSON Workflow:
    1. JSON → Parse articles → Existing ArticleProcessor pipeline
    
    PDF Workflow:  
    1. PDF → Mistral Document AI (text + structure extraction)
    2. Article Segmentation (identify individual articles)
    3. Metadata Enhancement (publication date, section, author)
    4. Quality Filtering (remove ads, ensure financial relevance)
    5. Feed to existing ArticleProcessor pipeline
    
    Both workflows end up in the same MongoDB collection for unified search.
    """
    
    def __init__(self, 
                 mistral_api_key: str,
                 mongo_connection_string: str,
                 database_name: str = "financial_research",
                 collection_name: str = "article_chunks"):
        """
        Initialize unified processor with support for both JSON and PDF files
        
        Args:
            mistral_api_key: Mistral AI API key
            mongo_connection_string: MongoDB connection string  
            database_name: MongoDB database name
            collection_name: Collection name for storing chunks
        """
        # Initialize Mistral client for Document AI
        self.mistral_client = MistralClient(api_key=mistral_api_key)
        
        # Initialize existing article processor
        self.article_processor = ArticleProcessor(
            mistral_api_key=mistral_api_key,
            mongo_connection_string=mongo_connection_string,
            database_name=database_name,
            collection_name=collection_name
        )
        
        # Document processing configuration
        self.supported_formats = ['.pdf', '.json']
        self.financial_keywords = [
            'federal reserve', 'fed', 'interest rate', 'inflation', 'market',
            'stock', 'bond', 'trading', 'investment', 'economy', 'gdp',
            'wall street', 'nasdaq', 'dow jones', 's&p 500', 'finance'
        ]
        
        print("✅ Initialized UnifiedDocumentProcessor")
        print("📄 Supported formats: JSON, PDF")
        print("🤖 Using Mistral Document AI for PDF extraction")
        print("📊 Using existing pipeline for JSON articles")
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Auto-detect file type based on extension
        
        Args:
            file_path: Path to file
            
        Returns:
            File type: 'json', 'pdf', or 'unsupported'
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.json':
            return 'json'
        elif file_extension == '.pdf':
            return 'pdf'
        else:
            return 'unsupported'
    
    def process_json_articles(self, json_file_path: str) -> Dict[str, int]:
        """
        Process JSON articles using existing ArticleProcessor
        
        Args:
            json_file_path: Path to JSON file containing articles
            
        Returns:
            Processing statistics
        """
        print(f"📊 Processing JSON file: {json_file_path}")
        return self.article_processor.process_articles_from_file(json_file_path)
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Unified file processing - auto-detects type and routes appropriately
        
        Args:
            file_path: Path to file (JSON or PDF)
            
        Returns:
            Processing results with unified format
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = self.detect_file_type(file_path)
        
        print(f"\n🔍 Detected file type: {file_type.upper()}")
        print(f"📁 Processing: {file_path}")
        
        if file_type == 'json':
            # Process JSON articles using existing pipeline
            json_stats = self.process_json_articles(file_path)
            
            # Convert to unified result format
            return {
                "success": True,
                "file_path": file_path,
                "file_type": "json",
                "processing_method": "existing_pipeline",
                "processing_stats": {
                    "total_articles": json_stats.get("total_articles", 0),
                    "successful_articles": json_stats.get("successful_articles", 0),
                    "total_chunks": json_stats.get("total_chunks", 0),
                    "successful_chunks": json_stats.get("successful_chunks", 0),
                    "failed_articles": json_stats.get("failed_articles", [])
                }
            }
            
        elif file_type == 'pdf':
            # Process PDF using Mistral Document AI
            pdf_result = self.process_pdf_newspaper(file_path)
            
            # Add file type info to result
            if pdf_result.get("success"):
                pdf_result["file_type"] = "pdf"
                pdf_result["processing_method"] = "mistral_document_ai"
            
            return pdf_result
            
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {file_type}",
                "supported_formats": self.supported_formats
            }
    
    def process_directory(self, directory_path: str, file_types: List[str] = None) -> Dict[str, Any]:
        """
        Process all supported files in a directory (JSON and/or PDF)
        
        Args:
            directory_path: Path to directory containing files
            file_types: Optional list to filter file types ['json', 'pdf']. If None, processes all.
            
        Returns:
            Aggregated processing results
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Default to all supported types
        if file_types is None:
            file_types = ['json', 'pdf']
        
        # Find all supported files
        all_files = []
        for file_type in file_types:
            if file_type == 'json':
                all_files.extend(dir_path.glob("*.json"))
            elif file_type == 'pdf':
                all_files.extend(dir_path.glob("*.pdf"))
        
        print(f"📁 Found {len(all_files)} files in {directory_path}")
        print(f"📄 File types: {file_types}")
        
        if not all_files:
            return {"success": False, "error": "No supported files found"}
        
        # Process each file
        results = {
            "directory": directory_path,
            "total_files": len(all_files),
            "successful_files": 0,
            "total_articles": 0,
            "total_chunks": 0,
            "failed_files": [],
            "file_breakdown": {"json": 0, "pdf": 0},
            "processing_details": []
        }
        
        for file_path in all_files:
            try:
                file_result = self.process_file(str(file_path))
                
                if file_result["success"]:
                    results["successful_files"] += 1
                    
                    # Add to totals
                    stats = file_result.get("processing_stats", {})
                    results["total_articles"] += stats.get("successful_articles", stats.get("articles_successfully_processed", 0))
                    results["total_chunks"] += stats.get("successful_chunks", stats.get("total_chunks_created", 0))
                    
                    # Track file type breakdown
                    file_type = file_result.get("file_type", "unknown")
                    if file_type in results["file_breakdown"]:
                        results["file_breakdown"][file_type] += 1
                else:
                    results["failed_files"].append(str(file_path))
                
                results["processing_details"].append(file_result)
                
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {str(e)}")
                results["failed_files"].append(str(file_path))
                continue
        
        print(f"\n🎯 UNIFIED BATCH PROCESSING COMPLETE!")
        print(f"📊 Files processed: {results['successful_files']}/{results['total_files']}")
        print(f"📄 File breakdown: JSON({results['file_breakdown']['json']}) + PDF({results['file_breakdown']['pdf']})")
        print(f"📰 Total articles: {results['total_articles']}")
        print(f"📦 Total chunks: {results['total_chunks']}")
        
        return results
    
    def encode_pdf_to_base64(self, pdf_path: str) -> str:
        """
        Encode PDF file to base64 for Mistral Document AI
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Base64 encoded PDF string
        """
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()
                base64_encoded = base64.b64encode(pdf_bytes).decode('utf-8')
            
            print(f"📄 Encoded PDF: {pdf_path} ({len(pdf_bytes)} bytes)")
            return base64_encoded
            
        except Exception as e:
            print(f"❌ Error encoding PDF {pdf_path}: {str(e)}")
            raise
    
    def extract_text_with_mistral_document_ai(self, pdf_base64: str, pdf_filename: str) -> Dict[str, Any]:
        """
        Extract structured text from PDF using Mistral Document AI
        
        Args:
            pdf_base64: Base64 encoded PDF
            pdf_filename: Original filename for context
            
        Returns:
            Structured extraction results from Mistral Document AI
        """
        try:
            # Construct prompt for newspaper-specific extraction
            prompt = f"""
            Analyze this newspaper PDF and extract structured information:
            
            EXTRACTION REQUIREMENTS:
            1. Identify individual articles (separate from ads, headers, footers)
            2. Extract article headlines and body text
            3. Determine publication date from newspaper header
            4. Identify sections (Business, Markets, Politics, etc.)
            5. Detect multi-column layout and preserve article boundaries
            6. Extract author bylines where available
            7. Focus on financial/economic content
            
            OUTPUT FORMAT (JSON):
            {{
                "publication_date": "YYYY-MM-DD",
                "newspaper_name": "name if identifiable",
                "total_pages": number,
                "articles": [
                    {{
                        "title": "article headline",
                        "content": "full article text",
                        "section": "Business/Markets/Politics/etc",
                        "author": "author name if available",
                        "page_number": page_number,
                        "column_position": "left/center/right",
                        "is_financial_content": true/false,
                        "confidence": 0.0-1.0
                    }}
                ]
            }}
            
            Focus on financial and economic articles. Exclude advertisements, classifieds, and non-relevant content.
            """
            
            # Call Mistral Document AI (using chat completion with document context)
            response = self.mistral_client.chat(
                model="mistral-large",  # Use largest model for complex document understanding
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "document", "document": {"data": pdf_base64, "type": "pdf"}}
                        ]
                    }
                ],
                max_tokens=4000,  # Large output for multiple articles
                temperature=0.1   # Low temperature for structured extraction
            )
            
            # Parse JSON response
            extraction_result = json.loads(response.choices[0].message.content)
            
            print(f"📰 Extracted {len(extraction_result.get('articles', []))} articles from {pdf_filename}")
            print(f"📅 Publication date: {extraction_result.get('publication_date', 'Unknown')}")
            
            return extraction_result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error for {pdf_filename}: {str(e)}")
            return {"articles": [], "error": "JSON parsing failed"}
        except Exception as e:
            print(f"❌ Mistral Document AI error for {pdf_filename}: {str(e)}")
            return {"articles": [], "error": str(e)}
    
    def filter_financial_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Filter articles to keep only financial/economic content
        
        Args:
            articles: List of extracted articles
            
        Returns:
            Filtered list of financial articles
        """
        financial_articles = []
        
        for article in articles:
            # Check if marked as financial content by AI
            if article.get('is_financial_content', False):
                financial_articles.append(article)
                continue
            
            # Fallback: keyword-based filtering
            title_lower = article.get('title', '').lower()
            content_lower = article.get('content', '').lower()
            section_lower = article.get('section', '').lower()
            
            # Check for financial keywords
            financial_score = 0
            for keyword in self.financial_keywords:
                if keyword in title_lower:
                    financial_score += 3  # Title keywords are more important
                if keyword in content_lower:
                    financial_score += 1
                if keyword in section_lower:
                    financial_score += 2
            
            # Check section
            financial_sections = ['business', 'markets', 'finance', 'economy', 'money']
            if any(section in section_lower for section in financial_sections):
                financial_score += 5
            
            # Keep if score is above threshold
            if financial_score >= 3:
                article['computed_financial_score'] = financial_score
                financial_articles.append(article)
        
        print(f"🔍 Filtered to {len(financial_articles)} financial articles")
        return financial_articles
    
    def convert_to_standard_format(self, pdf_extraction: Dict[str, Any], pdf_filename: str) -> List[Dict[str, Any]]:
        """
        Convert extracted PDF articles to standard JSON format for existing pipeline
        
        Args:
            pdf_extraction: Results from Mistral Document AI
            pdf_filename: Original PDF filename
            
        Returns:
            List of articles in standard format compatible with ArticleProcessor
        """
        standard_articles = []
        publication_date = pdf_extraction.get('publication_date', datetime.now().strftime('%Y-%m-%d'))
        newspaper_name = pdf_extraction.get('newspaper_name', 'Unknown Newspaper')
        
        for i, article in enumerate(pdf_extraction.get('articles', [])):
            # Generate unique article ID
            article_id = f"pdf_{Path(pdf_filename).stem}_{i:03d}"
            
            # Determine era based on publication date
            year = int(publication_date[:4]) if publication_date != 'Unknown' else datetime.now().year
            if year >= 2020:
                era = "current"
            elif year >= 2008:
                era = "crisis"
            elif year >= 1987:
                era = "greenspan"
            else:
                era = "volcker"
            
            # Create standard article format
            standard_article = {
                "id": article_id,
                "title": article.get('title', 'Untitled Article'),
                "content": article.get('content', ''),
                "source": newspaper_name,
                "published_date": f"{publication_date}T00:00:00Z",
                "category": "federal_reserve",  # Default category
                "era": era,
                "metadata": {
                    "author": article.get('author', ''),
                    "section": article.get('section', ''),
                    "page_number": article.get('page_number', 1),
                    "column_position": article.get('column_position', ''),
                    "extraction_confidence": article.get('confidence', 0.8),
                    "source_file": pdf_filename,
                    "extraction_method": "mistral_document_ai",
                    "processed_at": datetime.utcnow().isoformat(),
                    "financial_relevance_score": article.get('computed_financial_score', 0)
                }
            }
            
            standard_articles.append(standard_article)
        
        print(f"📋 Converted {len(standard_articles)} articles to standard format")
        return standard_articles
    
    def process_pdf_newspaper(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: PDF → extraction → filtering → standardization → chunking/embedding
        
        Args:
            pdf_path: Path to PDF newspaper file
            
        Returns:
            Processing statistics and results
        """
        print(f"\n🗞️  Processing PDF newspaper: {pdf_path}")
        
        # Validate file
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if Path(pdf_path).suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format. Supported: {self.supported_formats}")
        
        try:
            # Step 1: Encode PDF to base64
            pdf_base64 = self.encode_pdf_to_base64(pdf_path)
            
            # Step 2: Extract text and structure using Mistral Document AI
            extraction_result = self.extract_text_with_mistral_document_ai(
                pdf_base64, 
                Path(pdf_path).name
            )
            
            if 'error' in extraction_result:
                return {"success": False, "error": extraction_result['error']}
            
            # Step 3: Filter for financial content
            financial_articles = self.filter_financial_articles(extraction_result.get('articles', []))
            
            if not financial_articles:
                print("⚠️  No financial articles found in PDF")
                return {"success": False, "error": "No financial content detected"}
            
            # Step 4: Convert to standard format
            standard_articles = self.convert_to_standard_format(
                {**extraction_result, 'articles': financial_articles}, 
                Path(pdf_path).name
            )
            
            # Step 5: Process through existing pipeline (chunking + embedding + storage)
            total_chunks = 0
            successful_articles = 0
            
            for article in standard_articles:
                try:
                    chunks_processed = self.article_processor.process_single_article(article)
                    if chunks_processed > 0:
                        successful_articles += 1
                        total_chunks += chunks_processed
                except Exception as e:
                    print(f"❌ Failed to process article {article['id']}: {str(e)}")
                    continue
            
            # Prepare results
            results = {
                "success": True,
                "pdf_file": pdf_path,
                "extraction_metadata": {
                    "publication_date": extraction_result.get('publication_date'),
                    "newspaper_name": extraction_result.get('newspaper_name'),
                    "total_pages": extraction_result.get('total_pages')
                },
                "processing_stats": {
                    "raw_articles_extracted": len(extraction_result.get('articles', [])),
                    "financial_articles_filtered": len(financial_articles),
                    "articles_successfully_processed": successful_articles,
                    "total_chunks_created": total_chunks
                }
            }
            
            print(f"✅ Successfully processed PDF: {successful_articles} articles, {total_chunks} chunks")
            return results
            
        except Exception as e:
            print(f"❌ Error processing PDF {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process_pdf_directory(self, pdf_directory: str) -> Dict[str, Any]:
        """
        Process all PDF files in a directory
        
        Args:
            pdf_directory: Path to directory containing PDF files
            
        Returns:
            Aggregated processing results
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"📁 Found {len(pdf_files)} PDF files in {pdf_directory}")
        
        if not pdf_files:
            return {"success": False, "error": "No PDF files found"}
        
        # Process each PDF
        results = {
            "directory": pdf_directory,
            "total_pdfs": len(pdf_files),
            "successful_pdfs": 0,
            "total_articles": 0,
            "total_chunks": 0,
            "failed_pdfs": [],
            "processing_details": []
        }
        
        for pdf_file in pdf_files:
            try:
                pdf_result = self.process_pdf_newspaper(str(pdf_file))
                
                if pdf_result["success"]:
                    results["successful_pdfs"] += 1
                    results["total_articles"] += pdf_result["processing_stats"]["articles_successfully_processed"]
                    results["total_chunks"] += pdf_result["processing_stats"]["total_chunks_created"]
                else:
                    results["failed_pdfs"].append(str(pdf_file))
                
                results["processing_details"].append(pdf_result)
                
            except Exception as e:
                print(f"❌ Failed to process {pdf_file}: {str(e)}")
                results["failed_pdfs"].append(str(pdf_file))
                continue
        
        print(f"\n🎯 BATCH PROCESSING COMPLETE!")
        print(f"📊 PDFs processed: {results['successful_pdfs']}/{results['total_pdfs']}")
        print(f"📰 Articles processed: {results['total_articles']}")
        print(f"📦 Chunks created: {results['total_chunks']}")
        
        return results
    
    def close_connections(self):
        """Close all database connections"""
        self.article_processor.close_connections()
        print("🔌 Closed unified processor connections")

def main():
    """
    Demonstration of unified document processing (JSON + PDF)
    """
    # Configuration
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "your-mistral-api-key")
    MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "mongodb+srv://username:password@cluster.mongodb.net/")
    
    # Initialize unified processor
    unified_processor = UnifiedDocumentProcessor(
        mistral_api_key=MISTRAL_API_KEY,
        mongo_connection_string=MONGO_CONNECTION_STRING,
        database_name="financial_research",
        collection_name="article_chunks"
    )
    
    try:
        print("✅ Unified Document Processor ready for use!")
        print("\n📖 Usage examples:")
        print("   🔍 Auto-detect and process any file:")
        print("      processor.process_file('articles.json')     # JSON articles")
        print("      processor.process_file('newspaper.pdf')     # PDF newspaper")
        print()
        print("   📁 Process entire directories:")
        print("      processor.process_directory('./documents/')              # All JSON + PDF files")
        print("      processor.process_directory('./docs/', ['json'])         # Only JSON files")
        print("      processor.process_directory('./archive/', ['pdf'])       # Only PDF files")
        print()
        print("   📊 Specific file type processing:")
        print("      processor.process_json_articles('fed_articles.json')     # JSON only")
        print("      processor.process_pdf_newspaper('wsj_2024.pdf')          # PDF only")
        
        # Example usage (commented out):
        # 
        # # Process single file (auto-detects type)
        # result = unified_processor.process_file("fed_articles_dataset.json")
        # print(json.dumps(result, indent=2, default=str))
        #
        # # Process mixed directory  
        # batch_results = unified_processor.process_directory("./documents/")
        # print(json.dumps(batch_results, indent=2, default=str))
        
    finally:
        unified_processor.close_connections()

if __name__ == "__main__":
    main()