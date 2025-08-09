# ExploringDataLakes Backend

A FastAPI-based backend service for analyzing and clustering database tables using machine learning techniques. This service provides automated database documentation generation and intelligent table clustering based on semantic similarity.

## Features

- **Database File Upload**: Upload CSV files representing database tables
- **Ground Truth Management**: Upload and manage ground truth data for evaluation
- **Automated Documentation**: AI-powered generation of table and column descriptions
- **Semantic Clustering**: HDBSCAN-based clustering of database tables using sentence transformers
- **RESTful API**: Clean FastAPI endpoints for all functionality

## Architecture

The backend is organized into several key modules:

### Core Components

- **`app.py`**: Main FastAPI application with API endpoints
- **`Database.py`**: Central database management class for storing tables, ground truth, and descriptions
- **`helper.py`**: Shared instances and utilities (database, clustering, LLM clients)

### Compartmentalization Module

**Encoders**:
- `encoder.py`: Abstract base class for text encoders
- `sentence_transformer.py`: Sentence transformer implementation for encoding table descriptions

**Clusterors**:
- `clusteror.py`: Abstract base class for clustering algorithms
- `HDBScan.py`: HDBSCAN clustering implementation
- `Raptor.py`: Additional clustering implementation

### Description Generation

- **`descriptionGeneration.py`**: Core logic for generating table and column descriptions
- **`prompt.py`**: LLM prompt templates for documentation generation

## API Endpoints

### POST `/upload-database-file`
Upload a CSV file representing a database table.

**Parameters**:
- `file`: CSV file containing table data

**Response**:
```json
{
  "message": "Backend is working"
}
```

### POST `/upload-ground-truth`
Upload a JSON file containing ground truth data for evaluation.

**Parameters**:
- `file`: JSON file with ground truth data

**Response**:
```json
{
  "status": "success",
  "keys": ["key1", "key2", ...]
}
```

### GET `/HDBScanClustering`
Perform HDBSCAN clustering on uploaded database tables.

**Response**:
```json
{
  "1": {
    "0": ["table1: description", "table2: description"],
    "1": ["table3: description"],
    "-1": ["unclustered_table: description"]
  }
}
```

## Dependencies

### Core Dependencies
- **FastAPI** (^0.110): Modern web framework for building APIs
- **Uvicorn** (^0.29.0): ASGI server for running FastAPI
- **Pandas** (^2.3.1): Data manipulation and analysis
- **NumPy** (^2.3.2): Numerical computing

### Machine Learning
- **sentence-transformers** (^5.1.0): Text embedding models
- **hdbscan** (^0.8.40): Density-based clustering algorithm
- **umap** (^0.1.1): Dimensionality reduction

### AI Integration
- **OpenAI** (^1.99.5): LLM integration for description generation

### Utilities
- **python-multipart** (^0.0.20): File upload support

## Installation

### Using Poetry (Recommended)

1. Install Poetry if not already installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

### Using Docker

1. Build the Docker image:
```bash
docker build -t backend .
```

2. Run the container:
```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=your_api_key backend
```

## Configuration

### Environment Variables

- **`OPENAI_API_KEY`**: Required for AI-powered description generation
  ```bash
  export OPENAI_API_KEY="your-api-key-here"
  ```

## Usage

### Starting the Server

#### Development Mode
```bash
poetry run uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Production Mode
```bash
poetry run uvicorn app.app:app --host 0.0.0.0 --port 8000
```

### Basic Workflow

1. **Upload Database Tables**: Use the `/upload-database-file` endpoint to upload CSV files
2. **Upload Ground Truth** (Optional): Use `/upload-ground-truth` for evaluation data
3. **Generate Clusters**: Call `/HDBScanClustering` to analyze and cluster your tables

### Example Usage

```python
import requests

# Upload a database table
with open('employees.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload-database-file',
        files={'file': f}
    )

# Perform clustering
response = requests.get('http://localhost:8000/HDBScanClustering')
clusters = response.json()
```

## How It Works

### 1. Table Upload and Storage
CSV files are uploaded and stored as pandas DataFrames in the `Database` instance, indexed by filename.

### 2. Schema Extraction
The system analyzes each table to extract:
- Column data types
- Sample values (for context)
- Table structure

### 3. AI-Powered Documentation
Using the extracted schema and samples, an LLM generates structured JSON descriptions including:
- Table purpose and content
- Column descriptions and semantic meaning
- Data type classifications

### 4. Semantic Encoding
Table descriptions are encoded into high-dimensional vectors using sentence transformers, capturing semantic meaning and relationships.

### 5. Clustering
HDBSCAN algorithm groups tables based on semantic similarity, identifying:
- Related tables that could be part of the same domain
- Outlier tables that don't fit common patterns
- Hierarchical relationships between table groups

## Testing

Run the test suite:
```bash
poetry run pytest test/
```

Specific test modules:
```bash
# Test clustering functionality
poetry run pytest test/test_compartmentalization/

# Test description generation
poetry run pytest test/test_descriptionGeneration/
```

## Development

### Project Structure
```
backend/
├── app/
│   ├── app.py                          # Main FastAPI application
│   ├── Database.py                     # Core database management
│   ├── helper.py                       # Shared utilities
│   ├── compartmentalization/           # ML clustering modules
│   │   ├── encoders/                   # Text encoding implementations
│   │   └── clusterors/                 # Clustering algorithms
│   └── descriptionGeneration/          # AI documentation generation
├── test/                               # Test suite
├── Dockerfile                          # Container configuration
├── pyproject.toml                      # Poetry dependencies
└── README.md                           # This file
```

### Adding New Encoders

1. Create a new class inheriting from `Encoder`
2. Implement the `encode` method
3. Register in `helper.py`

### Adding New Clustering Algorithms

1. Create a new class inheriting from `Clusteror`
2. Implement the `cluster` method
3. Follow the return format: `{level: {clusterNo: [table names]}}`

## Contributing

1. Follow the existing code structure and patterns
2. Add tests for new functionality
3. Update documentation for API changes
4. Use type hints for better code maintainability

## License

This project is part of the ExploringDataLakes research initiative by Haseeb Muhammad.

## Support

For issues and questions, please refer to the main project repository or contact the development team.
