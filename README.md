# LLM-Powered Intelligent Query-Retrieval System

A comprehensive end-to-end system for processing large documents and making contextual decisions in insurance, legal, HR, and compliance domains. Built with FastAPI, PostgreSQL, and advanced AI technologies.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   LLM Parser     │    │   Embedding     │
│   Processing    │───▶│   (GPT-4)        │───▶│   Generation    │
│   (PDF/DOCX)    │    │                  │    │   (OpenAI)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Vector DB     │    │   Clause         │    │   Decision      │
│   (FAISS/Pine-  │───▶│   Matching       │───▶│   Engine        │
│   cone)         │    │   (Semantic)     │    │   (GPT-4)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   FastAPI        │    │   JSON          │
│   Database      │    │   REST API       │    │   Response      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Features

- **Multi-format Document Processing**: PDF, DOCX, and email support
- **Advanced NLP**: GPT-4 powered query parsing and answer generation
- **Semantic Search**: Vector-based retrieval using FAISS or Pinecone
- **Explainable AI**: Detailed reasoning and confidence scoring
- **Scalable Architecture**: Async FastAPI with database optimization
- **Production Ready**: Docker containerization and comprehensive monitoring

## 📋 Requirements

- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key
- PostgreSQL 15+
- 4GB+ RAM (for local embedding models)

## 🛠️ Installation & Setup

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd llm-query-retrieval-system
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file with your OpenAI API key and other configurations
```

3. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

4. **Access the application**
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Database Admin: http://localhost:5050 (pgAdmin)

### Option 2: Local Development

1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up PostgreSQL**
```bash
# Install PostgreSQL and create database
createdb llm_retrieval
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file with your configurations
```

4. **Run the application**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 and embeddings | Yes | - |
| `DATABASE_URL` | PostgreSQL connection string | Yes | `postgresql://...` |
| `VECTOR_DB_TYPE` | Vector database type (`faiss` or `pinecone`) | No | `faiss` |
| `HACKRX_TOKEN` | Authentication token for HackRX API | Yes | (provided) |
| `DEBUG` | Enable debug mode | No | `false` |
| `CHUNK_SIZE` | Document chunk size for processing | No | `1000` |
| `MAX_TOKENS` | Maximum LLM response tokens | No | `4000` |

### Vector Database Options

**FAISS (Local, Recommended for Development)**
- No external dependencies
- Fast local search
- Persistent storage to disk

**Pinecone (Cloud, Recommended for Production)**
- Managed vector database
- Horizontal scaling
- Real-time updates

## 📡 API Usage

### HackRX Endpoint

The main endpoint matching the problem specification:

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 2a91272cc18b579a54b1281caf08e0c883f1daf70cbafa6418ca3778fbc17df3" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?",
      "Does this policy cover maternity expenses?"
    ]
  }'
```

### Response Format

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment...",
    "There is a waiting period of thirty-six (36) months...",
    "Yes, the policy covers maternity expenses with conditions..."
  ]
}
```

### Additional Endpoints

- `GET /health` - System health check
- `POST /process-document` - Process individual documents
- `GET /docs` - Interactive API documentation

## 🏃‍♂️ Running the System

### Development Mode

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Production Mode

```bash
# Use production profile with Nginx
docker-compose --profile production up -d

# Scale the application
docker-compose up --scale app=3
```

## 🧪 Testing

### Run Test Suite

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v --cov=app

# Run specific test categories
pytest tests/test_document_processor.py -v
pytest tests/test_llm_service.py -v
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Test document processing
curl -X POST "http://localhost:8000/process-document" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_url": "https://example.com/document.pdf"}'
```

## 📊 Monitoring & Logging

### Application Logs

```bash
# View application logs
docker-compose logs -f app

# Log files location
tail -f logs/app.log
```

### Database Monitoring

- **pgAdmin**: http://localhost:5050
  - Email: admin@example.com
  - Password: admin

### Performance Metrics

The system tracks:
- Query processing time
- Token usage optimization
- Cache hit rates
- Error rates and types
- Database query performance

## 🔍 System Components

### Document Processor (`app/services/document_processor.py`)
- Multi-format document parsing
- Intelligent text chunking
- Metadata extraction
- Error handling and validation

### Embedding Service (`app/services/embedding_service.py`)
- OpenAI embeddings with local fallback
- Batch processing optimization
- Caching mechanism
- Rate limiting

### LLM Service (`app/services/llm_service.py`)
- Query parsing and intent detection
- Context-aware answer generation
- Confidence scoring
- Explainable reasoning

### Vector Search Service (`app/services/vector_search_service.py`)
- Dual FAISS/Pinecone support
- Semantic similarity search
- Index management
- Performance optimization

## 🛡️ Security

- Bearer token authentication
- Input validation and sanitization
- SQL injection protection
- Rate limiting
- Secure configuration management

## 📈 Performance Optimization

### Token Efficiency
- Intelligent context truncation
- Batch processing for embeddings
- Caching frequent queries
- Optimized prompt engineering

### Latency Optimization
- Async processing throughout
- Connection pooling
- Vector index optimization
- Smart chunking strategies

## 🚀 Deployment

### Docker Production Deployment

```bash
# Build production image
docker build -t llm-retrieval:latest .

# Run with production settings
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e DATABASE_URL=your_db_url \
  llm-retrieval:latest
```

### Cloud Deployment

The system is designed to deploy on:
- AWS ECS/EKS
- Google Cloud Run
- Azure Container Instances
- Any Kubernetes cluster

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Ensure `OPENAI_API_KEY` is set in your `.env` file
- Verify the key is valid and has sufficient credits

**"Database connection failed"**
- Check PostgreSQL is running: `docker-compose ps`
- Verify `DATABASE_URL` in environment variables
- Ensure database exists: `docker-compose exec postgres psql -U postgres -l`

**"Vector index not found"**
- FAISS index is created automatically on first document processing
- For Pinecone, ensure valid API key and environment are set

**"Document processing timeout"**
- Large documents may take time to process
- Increase timeout in configuration
- Check document format is supported

### Getting Help

- Check the `/health` endpoint for system status
- Review logs in `logs/app.log`
- Enable debug mode: `DEBUG=true`
- Check API documentation at `/docs`

## 🎯 Future Enhancements

- Multi-language document support
- Advanced OCR for scanned documents
- Real-time document collaboration
- Enhanced security with OAuth2
- Machine learning model fine-tuning
- Advanced analytics dashboard

---

**Built with ❤️ for intelligent document processing**