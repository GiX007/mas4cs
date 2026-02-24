# Ollama Basics

## Check installation
```bash
ollama --version
```

## See installed models
```bash
ollama list
```

## Download a model
```bash
ollama pull model:tag
```

**Examples:**
```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
```

## Run a model
```bash
ollama run model:tag
```

**Example:**
```bash
ollama run mistral:7b
```

## Delete a model (frees disk)
```bash
ollama rm model:tag
```

## Update a model
Re-run the **pull** command.

## See models currently loaded in RAM
```bash
ollama ps
```

## Stop a running model
Press **CTRL + C**

---
