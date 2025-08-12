# Deployment Guide

## Setting up OpenAI API Key

### Option 1: Environment File (Recommended)

1. Create a `.env` file in the root directory:
```bash
# .env
OPENAI_API_KEY=your-actual-openai-api-key-here
```

2. Make sure the `.env` file is in your `.gitignore` to keep it secure:
```bash
echo ".env" >> .gitignore
```

3. Deploy using Docker Compose:
```bash
docker-compose up --build
```

### Option 2: Direct Environment Variable

Set the environment variable directly:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
docker-compose up --build
```

### Option 3: Cloud Platform Deployment

#### For AWS, Google Cloud, or Azure:

Set the environment variable in your cloud platform's configuration:

- **AWS ECS**: Add to task definition environment variables
- **Google Cloud Run**: Set via `--set-env-vars` flag
- **Azure Container Instances**: Add to environment variables section

#### For Heroku:

```bash
heroku config:set OPENAI_API_KEY=your-openai-api-key-here
```

#### For Railway:

Add the environment variable in the Railway dashboard or via CLI:

```bash
railway variables set OPENAI_API_KEY=your-openai-api-key-here
```

### Option 4: Kubernetes Deployment

Add the environment variable to your deployment YAML:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  template:
    spec:
      containers:
      - name: backend
        image: your-backend-image
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

Create the secret:
```bash
kubectl create secret generic openai-secret --from-literal=api-key=your-openai-api-key-here
```

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables or secrets management**
3. **Rotate API keys regularly**
4. **Use least privilege access**
5. **Monitor API usage and costs**

## Verification

After deployment, check that the API key is working by:

1. Looking at the backend logs for the success message
2. Testing the description generation endpoint
3. Verifying no "Warning: OPENAI_API_KEY not set" messages appear

## Troubleshooting

- If you see "LLM functionality will be disabled", the API key is not properly set
- Check that the environment variable name is exactly `OPENAI_API_KEY`
- Ensure the API key is valid and has sufficient credits
- Verify the `.env` file is in the correct location (root directory)
