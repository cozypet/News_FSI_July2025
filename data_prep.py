"""
Article Processing Pipeline for Vector Search
Chunks articles with recursive chunker, generates embeddings with Mistral, and stores in MongoDB
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()

# Core dependencies
import pymongo
from pymongo import MongoClient
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

@dataclass
class ArticleChunk:
    """Data class for article chunks with metadata"""
    chunk_id: str
    article_id: str
    content: str
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None

class ArticleProcessor:
    """Process articles for vector search preparation"""
    
    def __init__(self, 
                 mistral_api_key: str,
                 mongo_connection_string: str,
                 database_name: str = "financial_research",
                 collection_name: str = "article_chunks"):
        """
        Initialize the processor with API keys and database configuration
        
        Args:
            mistral_api_key: Mistral AI API key
            mongo_connection_string: MongoDB connection string
            database_name: MongoDB database name
            collection_name: MongoDB collection name for storing chunks
        """
        # Initialize Mistral client
        self.mistral_client = MistralClient(api_key=mistral_api_key)
        
        # Initialize MongoDB client
        self.mongo_client = MongoClient(mongo_connection_string)
        self.db = self.mongo_client[database_name]
        self.collection = self.db[collection_name]
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Target chunk size in characters
            chunk_overlap=200,  # Overlap between chunks for context preservation
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Split hierarchy
        )
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print(f"✅ Initialized ArticleProcessor")
        print(f"📊 Database: {database_name}")
        print(f"📚 Collection: {collection_name}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def chunk_article(self, article: Dict[str, Any]) -> List[ArticleChunk]:
        """
        Chunk a single article using recursive character text splitter
        
        Args:
            article: Article dictionary with content and metadata
            
        Returns:
            List of ArticleChunk objects
        """
        # Combine title and content for chunking
        full_text = f"Title: {article['title']}\n\nContent: {article['content']}"
        
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(full_text)
        
        # Create ArticleChunk objects
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = ArticleChunk(
                chunk_id=f"{article['id']}_chunk_{i:03d}",
                article_id=article['id'],
                content=chunk_text,
                chunk_index=i,
                total_chunks=len(text_chunks),
                metadata={
                    "title": article['title'],
                    "source": article['source'],
                    "published_date": article['published_date'],
                    "category": article['category'],
                    "era": article['era'],
                    "tags": article.get('tags', []),
                    "author": article.get('metadata', {}).get('author', ''),
                    "word_count": article.get('metadata', {}).get('word_count', 0),
                    "token_count": self.count_tokens(chunk_text),
                    "processed_at": datetime.utcnow().isoformat()
                }
            )
            chunks.append(chunk)
        
        print(f"📄 Chunked article '{article['id']}' into {len(chunks)} chunks")
        return chunks
    
    def generate_embedding(self, text: str, model: str = "mistral-embed") -> List[float]:
        """
        Generate embedding for text using Mistral AI
        
        Args:
            text: Text to embed
            model: Mistral embedding model to use
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Generate embedding using Mistral
            response = self.mistral_client.embeddings(
                model=model,
                input=[text]
            )
            
            embedding = response.data[0].embedding
            print(f"🔢 Generated embedding: {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {str(e)}")
            raise
    
    def store_chunk_in_mongodb(self, chunk: ArticleChunk) -> bool:
        """
        Store a single chunk with its embedding in MongoDB
        
        Args:
            chunk: ArticleChunk object to store
            
        Returns:
            Boolean indicating success
        """
        try:
            # Prepare document for MongoDB
            document = {
                "_id": chunk.chunk_id,
                "article_id": chunk.article_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "embedding": chunk.embedding,
                "metadata": chunk.metadata
            }
            
            # Insert into MongoDB (upsert to handle duplicates)
            result = self.collection.replace_one(
                {"_id": chunk.chunk_id}, 
                document, 
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                print(f"💾 Stored chunk: {chunk.chunk_id}")
                return True
            else:
                print(f"⚠️  Chunk already exists: {chunk.chunk_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error storing chunk {chunk.chunk_id}: {str(e)}")
            return False
    
    def create_vector_search_index(self, index_name: str = "vector_search_index"):
        """
        Create vector search index in MongoDB Atlas
        Note: This requires MongoDB Atlas with vector search enabled
        """
        try:
            # Vector search index definition
            index_definition = {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024,  # Mistral embedding dimension
                        "similarity": "cosine"
                    },
                    {
                        "type": "filter",
                        "path": "metadata.era"
                    },
                    {
                        "type": "filter", 
                        "path": "metadata.category"
                    },
                    {
                        "type": "filter",
                        "path": "metadata.tags"
                    }
                ]
            }
            
            print(f"🔍 Creating vector search index: {index_name}")
            print("Note: This must be done through MongoDB Atlas UI or Atlas Admin API")
            print(f"Index definition: {json.dumps(index_definition, indent=2)}")
            
        except Exception as e:
            print(f"❌ Error creating index: {str(e)}")
    
    def process_single_article(self, article: Dict[str, Any]) -> int:
        """
        Process a single article: chunk -> embed -> store
        
        Args:
            article: Article dictionary
            
        Returns:
            Number of chunks successfully processed
        """
        print(f"\n🔄 Processing article: {article['id']}")
        
        # Step 1: Chunk the article
        chunks = self.chunk_article(article)
        
        # Step 2: Generate embeddings and store chunks
        successful_chunks = 0
        for chunk in chunks:
            try:
                # Generate embedding
                chunk.embedding = self.generate_embedding(chunk.content)
                
                # Store in MongoDB
                if self.store_chunk_in_mongodb(chunk):
                    successful_chunks += 1
                    
            except Exception as e:
                print(f"❌ Failed to process chunk {chunk.chunk_id}: {str(e)}")
                continue
        
        print(f"✅ Successfully processed {successful_chunks}/{len(chunks)} chunks for article {article['id']}")
        return successful_chunks
    
    def process_articles_from_file(self, json_file_path: str) -> Dict[str, int]:
        """
        Process all articles from a JSON file
        
        Args:
            json_file_path: Path to JSON file containing articles
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"📂 Loading articles from: {json_file_path}")
        
        # Load articles from JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        if isinstance(articles, dict) and 'articles' in articles:
            articles = articles['articles']  # Handle wrapped format
        
        print(f"📊 Found {len(articles)} articles to process")
        
        # Process each article
        stats = {
            "total_articles": len(articles),
            "successful_articles": 0,
            "total_chunks": 0,
            "successful_chunks": 0,
            "failed_articles": []
        }
        
        for i, article in enumerate(articles, 1):
            try:
                print(f"\n{'='*60}")
                print(f"📰 Article {i}/{len(articles)}: {article.get('title', 'Unknown')[:50]}...")
                
                chunks_processed = self.process_single_article(article)
                
                if chunks_processed > 0:
                    stats["successful_articles"] += 1
                    stats["successful_chunks"] += chunks_processed
                else:
                    stats["failed_articles"].append(article['id'])
                    
                stats["total_chunks"] += len(self.chunk_article(article))
                
            except Exception as e:
                print(f"❌ Failed to process article {article.get('id', 'unknown')}: {str(e)}")
                stats["failed_articles"].append(article.get('id', 'unknown'))
                continue
        
        # Print final statistics
        print(f"\n{'='*60}")
        print("🎯 PROCESSING COMPLETE!")
        print(f"📊 Articles processed: {stats['successful_articles']}/{stats['total_articles']}")
        print(f"📦 Chunks processed: {stats['successful_chunks']}/{stats['total_chunks']}")
        if stats['failed_articles']:
            print(f"❌ Failed articles: {stats['failed_articles']}")
        
        return stats
    
    def search_similar_chunks(self, query_text: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar chunks using vector search
        Note: Requires vector search index to be created in MongoDB Atlas
        
        Args:
            query_text: Text to search for
            limit: Number of results to return
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query_text)
            
            # Vector search pipeline (requires MongoDB Atlas vector search)
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_search_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "article_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            print(f"🔍 Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            return []
    
    def close_connections(self):
        """Close database connections"""
        self.mongo_client.close()
        print("🔌 Closed database connections")

def main():
    """
    Main function to demonstrate usage
    """
    # Configuration - replace with your actual values
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "your-mistral-api-key")
    MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING", "mongodb+srv://username:password@cluster.mongodb.net/")
    
    # Initialize processor
    processor = ArticleProcessor(
        mistral_api_key=MISTRAL_API_KEY,
        mongo_connection_string=MONGO_CONNECTION_STRING,
        database_name="financial_research",
        collection_name="article_chunks"
    )
    
    try:
        # Process articles from JSON file
        stats = processor.process_articles_from_file("fed_articles_dataset.json")
        
        # Create vector search index (instructions)
        processor.create_vector_search_index()
        
        # Example search (requires vector search index)
        # results = processor.search_similar_chunks("Fed rate hike market reaction", limit=3)
        # print(f"Search results: {json.dumps(results, indent=2, default=str)}")
        
    finally:
        # Clean up
        processor.close_connections()

if __name__ == "__main__":
    main()