# Multi-Language Knowledge Graph Construction

AutoSchemaKG provides comprehensive support for constructing knowledge graphs in multiple languages. This guide explains how to configure and use the multi-language capabilities for triple extraction and concept generation.

## Overview

The multi-language system is based on three key components:

1. **Language keys in prompts**: Different extraction instructions for each language
2. **Metadata in corpus**: Each document specifies its language
3. **Language-specific concept generation**: Concepts generated based on specified language

## Language Configuration

### 1. Multi-Language Prompt Structure

Create a prompt file with language-specific instructions for each language you want to support:

```json
{
  "en": {
    "system": "You are a helpful assistant",
    "triple_extraction": "Extract knowledge graph triples from English text..."
  },
  "zh-CN": {
    "system": "你是一个有用的助手",
    "triple_extraction": "从简体中文文本中提取知识图谱三元组..."
  },
  "zh-HK": {
    "system": "你是一個有用的助手",
    "triple_extraction": "從繁體中文文本中提取知識圖譜三元組..."
  },
  "ja": {
    "system": "あなたは役立つアシスタントです",
    "triple_extraction": "日本語テキストから知識グラフのトリプルを抽出します..."
  }
}
```

**Supported Language Codes:**
- `en`: English
- `zh-CN`: Simplified Chinese (China)
- `zh-HK`: Traditional Chinese (Hong Kong)
- `zh-TW`: Traditional Chinese (Taiwan)
- `ja`: Japanese
- `ko`: Korean
- `fr`: French
- `de`: German
- `es`: Spanish
- `ru`: Russian
- `ar`: Arabic
- `hi`: Hindi
- And any other ISO 639-1 or locale code you define

### 2. Corpus Data Format with Language Metadata

Each document in your corpus **must** include language metadata to enable automatic language detection:

```json
[
    {
        "id": "1",
        "text": "The quick brown fox jumps over the lazy dog.",
        "metadata": {
            "lang": "en"
        }
    },
    {
        "id": "2",
        "text": "话说天下大势，分久必合，合久必分。",
        "metadata": {
            "lang": "zh-CN"
        }
    },
    {
        "id": "3",
        "text": "話說天下大勢，分久必合，合久必分。",
        "metadata": {
            "lang": "zh-HK"
        }
    },
    {
        "id": "4",
        "text": "吾輩は猫である。名前はまだ無い。",
        "metadata": {
            "lang": "ja"
        }
    }
]
```

**Required Fields:**
- `id`: Unique document identifier (string or number)
- `text`: Document content in the specified language
- `metadata.lang`: Language code that matches your prompt keys

## Complete Multi-Language Extraction Example

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI

# Initialize LLM client (supports vLLM, SGLang, OpenAI, etc.)
client = OpenAI(
    base_url="http://localhost:8135/v1",
    api_key="EMPTY"
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
triple_generator = LLMGenerator(client, model_name=model_name)

# Configure with multi-language prompts
kg_extraction_config = ProcessingConfig(
    model_path=model_name,
    data_directory="example_data/multilingual_data",
    filename_pattern="RomanceOfTheThreeKingdom",
    batch_size_triple=16,
    batch_size_concept=64,
    output_directory=f"generated/RomanceOfTheThreeKingdom",
    
    # Specify custom multi-language prompts
    triple_extraction_prompt_path="custom_prompts/multilingual_prompt.json",
    triple_extraction_schema_path="custom_prompts/custom_schema.json",
    
    max_new_tokens=8192,
    record=True
)

kg_extractor = KnowledgeGraphExtractor(
    model=triple_generator, 
    config=kg_extraction_config
)

# Step 1: Extract triples (automatically uses language from metadata)
kg_extractor.run_extraction()

# Step 2: Convert to CSV
kg_extractor.convert_json_to_csv()

# Step 3: Generate concepts for Simplified Chinese
kg_extractor.generate_concept_csv_temp(language='zh-CN')

# Or for Traditional Chinese (Hong Kong)
# kg_extractor.generate_concept_csv_temp(language='zh-HK')

# Or for English
# kg_extractor.generate_concept_csv_temp(language='en')

# Step 4: Create concept CSV files
kg_extractor.create_concept_csv()

# Step 5: Convert to GraphML
kg_extractor.convert_to_graphml()
```

## How Language Matching Works

### Triple Extraction Phase

The system automatically handles language selection during extraction:

1. **Document Processing**: System reads `metadata.lang` from each document
2. **Prompt Matching**: Matches the language code with prompt keys (e.g., `"zh-CN"`)
3. **Instruction Selection**: Uses the corresponding language-specific extraction instructions
4. **Fallback Mechanism**: If no matching key is found, falls back to `"en"` or the first available language

**Example Flow:**
```
Document: {"id": "1", "text": "今天天气很好", "metadata": {"lang": "zh-CN"}}
         ↓
Prompt Key: "zh-CN" → Uses Chinese extraction instructions
         ↓
Extracted Triples: [{"subject": "今天", "relation": "天气", "object": "很好"}]
```

### Concept Generation Phase

Concept generation requires **explicit language specification**:

```python
# You must specify which language to use for concept generation
kg_extractor.generate_concept_csv_temp(language='zh-CN')
```

The system will:
1. Use the specified language's concept generation prompts
2. Generate concepts using that language's vocabulary and grammar
3. All concepts will be in the specified language

**Why Explicit?** Concept generation is a corpus-level operation, so you need to decide which language to use for conceptualization.

## Multi-Language Project Structure

Organize your multi-language projects as follows:

```
my_multilingual_project/
├── data/
│   ├── english_corpus.json          # metadata.lang = "en"
│   ├── chinese_simplified.json      # metadata.lang = "zh-CN"
│   ├── chinese_traditional.json     # metadata.lang = "zh-HK"
│   └── japanese_corpus.json         # metadata.lang = "ja"
├── prompts/
│   ├── multilingual_prompt.json     # All languages in one file
│   │   # {"en": {...}, "zh-CN": {...}, "zh-HK": {...}, "ja": {...}}
│   └── custom_schema.json           # Language-agnostic schema
├── output/
│   ├── english_kg/
│   │   ├── kg_extraction/
│   │   ├── concepts/
│   │   └── kg_graphml/
│   ├── chinese_simplified_kg/
│   │   ├── kg_extraction/
│   │   ├── concepts/
│   │   └── kg_graphml/
│   └── chinese_traditional_kg/
│       ├── kg_extraction/
│       ├── concepts/
│       └── kg_graphml/
└── scripts/
    ├── extract_english.py
    ├── extract_chinese.py
    └── extract_all.sh
```

## Best Practices

### 1. Use Standard Language Codes

Always use ISO 639-1 codes or standard locale codes:

✅ **Good:**
- `en`, `zh-CN`, `zh-HK`, `ja`, `ko`, `fr-FR`, `es-ES`

❌ **Bad:**
- `chinese`, `english`, `中文`, `japanese_text`

### 2. Create Language-Specific Prompts

Tailor extraction instructions to each language's unique characteristics:

```json
{
  "en": {
    "system": "You are a knowledge graph expert.",
    "triple_extraction": "Extract entities and relationships. Use clear, concise relation names."
  },
  "zh-CN": {
    "system": "你是一个知识图谱专家。",
    "triple_extraction": "提取实体和关系。注意处理中文分词，保持关系名称简洁明确。"
  },
  "ja": {
    "system": "あなたは知識グラフの専門家です。",
    "triple_extraction": "エンティティと関係を抽出します。日本語の助詞に注意し、明確な関係名を使用してください。"
  }
}
```

**Language-Specific Considerations:**

- **Chinese (zh-CN, zh-HK)**: Mention word segmentation, use proper punctuation
- **Japanese (ja)**: Consider particles (は、が、を), kanji vs hiragana
- **Korean (ko)**: Note subject/object markers, formal vs informal
- **Arabic (ar)**: Right-to-left text, diacritics, verb forms
- **German (de)**: Compound words, capitalization rules

### 3. Separate Concept Generation by Language

For mixed-language corpora, generate concepts separately for each language:

```python
# Process English documents
kg_extractor_en = KnowledgeGraphExtractor(model=triple_generator, config=config_en)
kg_extractor_en.run_extraction()
kg_extractor_en.convert_json_to_csv()
kg_extractor_en.generate_concept_csv_temp(language='en')
kg_extractor_en.create_concept_csv()

# Process Chinese documents
kg_extractor_zh = KnowledgeGraphExtractor(model=triple_generator, config=config_zh)
kg_extractor_zh.run_extraction()
kg_extractor_zh.convert_json_to_csv()
kg_extractor_zh.generate_concept_csv_temp(language='zh-CN')
kg_extractor_zh.create_concept_csv()
```

### 4. Validate Metadata Before Processing

Ensure all documents have valid language metadata:

```python
import json

def validate_corpus(file_path):
    """Validate that all documents have proper language metadata."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    errors = []
    for doc in data:
        # Check for id
        if 'id' not in doc:
            errors.append(f"Document missing 'id': {doc.get('text', '')[:50]}...")
        
        # Check for metadata
        if 'metadata' not in doc:
            errors.append(f"Document {doc.get('id', 'unknown')} missing 'metadata'")
        elif 'lang' not in doc['metadata']:
            errors.append(f"Document {doc.get('id', 'unknown')} missing 'lang' in metadata")
        
        # Check for text
        if 'text' not in doc or not doc['text'].strip():
            errors.append(f"Document {doc.get('id', 'unknown')} has empty text")
    
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"✓ All {len(data)} documents are valid")
        return True

# Usage
validate_corpus("example_data/multilingual_data/corpus.json")
```

### 5. Choose Appropriate LLM Models

Different models have different language capabilities:

**English-Focused Models:**
- GPT-4, GPT-3.5-turbo
- Llama-3, Llama-3.1, Llama-3.3
- Mistral, Mixtral

**Chinese-Focused Models:**
- Qwen/Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct
- ChatGLM, ChatGLM2, ChatGLM3
- Baichuan-7B, Baichuan-13B
- Yi-6B, Yi-34B

**Multilingual Models:**
- BLOOM, BLOOMZ
- mT5, mT0
- XGLM
- Qwen (supports 29+ languages)

**Recommendation:** For best results, use models trained on the target language(s).

## Example: Processing Traditional Chinese Text

Here's a complete example using the Romance of the Three Kingdoms dataset:

### Step 1: Prepare Data

```json
// example_data/multilingual_data/RomanceOfTheThreeKingdom-zh-HK.json
[
    {
        "id": 1,
        "text": "話說天下大勢，分久必合，合久必分。周末七國分爭，并入於秦...",
        "metadata": {
            "lang": "zh-HK"
        }
    },
    {
        "id": 2,
        "text": "建寧二年四月望日，帝御溫德殿。方升座，殿角狂風驟起...",
        "metadata": {
            "lang": "zh-HK"
        }
    }
]
```

### Step 2: Create Chinese Prompt

```json
// custom_prompts/chinese_prompt.json
{
  "zh-HK": {
    "system": "你是一個專業的知識圖譜構建助手，精通繁體中文文本分析。",
    "triple_extraction": "從以下繁體中文文本中提取知識圖譜三元組。每個三元組包含主體、關係和客體。請確保：\n1. 主體和客體是具體的實體或概念\n2. 關係描述清晰準確\n3. 提取所有重要的事實信息\n4. 輸出格式為JSON數組"
  }
}
```

### Step 3: Run Extraction

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8135/v1",
    api_key="EMPTY"
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
triple_generator = LLMGenerator(client, model_name=model_name)

# Configure for Traditional Chinese
kg_extraction_config = ProcessingConfig(
    model_path=model_name,
    data_directory="example_data/multilingual_data",
    filename_pattern="RomanceOfTheThreeKingdom-zh-HK",
    triple_extraction_prompt_path="custom_prompts/chinese_prompt.json",
    output_directory="generated/RomanceOfTheThreeKingdom_HK",
    batch_size_triple=16,
    batch_size_concept=64,
    max_new_tokens=8192,
    record=True
)

# Create extractor
kg_extractor = KnowledgeGraphExtractor(
    model=triple_generator, 
    config=kg_extraction_config
)

# Extract triples (uses zh-HK prompts automatically)
print("Step 1: Extracting triples...")
kg_extractor.run_extraction()

# Convert to CSV
print("Step 2: Converting to CSV...")
kg_extractor.convert_json_to_csv()

# Generate Traditional Chinese concepts
print("Step 3: Generating concepts...")
kg_extractor.generate_concept_csv_temp(language='zh-HK')

# Create concept CSV files
print("Step 4: Creating concept CSV...")
kg_extractor.create_concept_csv()

# Convert to GraphML
print("Step 5: Converting to GraphML...")
kg_extractor.convert_to_graphml()

print("✓ Knowledge graph construction complete!")
```

### Step 4: Verify Output

```python
import json

# Check extracted triples
with open("generated/RomanceOfTheThreeKingdom_HK/kg_extraction/RomanceOfTheThreeKingdom-zh-HK_1_in_1.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f"Extracted {len(data)} documents")
    if data:
        print(f"Sample triples from first document:")
        for triple in data[0]['triples'][:3]:
            print(f"  {triple['subject']} --[{triple['relation']}]--> {triple['object']}")
```

## Mixed-Language Corpus Handling

### Scenario: Documents in Multiple Languages

If you have a corpus with documents in different languages:

```json
[
    {"id": "1", "text": "Albert Einstein was a physicist.", "metadata": {"lang": "en"}},
    {"id": "2", "text": "爱因斯坦是一位物理学家。", "metadata": {"lang": "zh-CN"}},
    {"id": "3", "text": "アインシュタインは物理学者でした。", "metadata": {"lang": "ja"}}
]
```

**Processing Strategy:**

```python
# Option 1: Process all at once (requires multilingual prompt)
kg_extraction_config = ProcessingConfig(
    data_directory="mixed_corpus",
    triple_extraction_prompt_path="prompts/multilingual_prompt.json",  # Has en, zh-CN, ja keys
    # ... other config
)
kg_extractor.run_extraction()  # Automatically routes to correct language

# Option 2: Split by language first
import json

with open("mixed_corpus/data.json") as f:
    docs = json.load(f)

# Split by language
en_docs = [d for d in docs if d['metadata']['lang'] == 'en']
zh_docs = [d for d in docs if d['metadata']['lang'] == 'zh-CN']
ja_docs = [d for d in docs if d['metadata']['lang'] == 'ja']

# Save separately and process
with open("en_corpus.json", 'w') as f:
    json.dump(en_docs, f, ensure_ascii=False, indent=2)
# Then process each separately...
```

## Troubleshooting

### Issue: Wrong Language Prompts Used

**Symptom:** Extraction uses English prompts for Chinese text

**Solution:** 
1. Check `metadata.lang` matches prompt keys exactly
2. Verify JSON prompt file is valid (no trailing commas)
3. Ensure prompt file path is correct

```python
# Debug: Check what language is detected
import json
with open("your_corpus.json") as f:
    docs = json.load(f)
    langs = set(d['metadata']['lang'] for d in docs if 'metadata' in d and 'lang' in d['metadata'])
    print(f"Languages in corpus: {langs}")

with open("your_prompt.json") as f:
    prompts = json.load(f)
    print(f"Languages in prompts: {set(prompts.keys())}")
```

### Issue: Concepts Generated in Wrong Language

**Symptom:** Asked for Chinese concepts but got English

**Solution:** Double-check the language parameter:

```python
# Wrong
kg_extractor.generate_concept_csv_temp(language='cn')  # Should be 'zh-CN'

# Correct
kg_extractor.generate_concept_csv_temp(language='zh-CN')
```

### Issue: Mixed Language in Output

**Symptom:** Some triples in English, some in Chinese

**Causes:**
1. Model mixing languages (use better model)
2. Source documents have mixed language content
3. Prompt not clear enough about language consistency

**Solutions:**
- Use language-specific models
- Add explicit language instructions in prompts
- Clean source data to ensure language purity

## Advanced: Cross-Language Knowledge Graphs

For building knowledge graphs that link concepts across languages:

```python
# Step 1: Extract in each language separately
configs = {
    'en': ProcessingConfig(data_directory="data/en", output_directory="output/en", ...),
    'zh-CN': ProcessingConfig(data_directory="data/zh", output_directory="output/zh", ...),
    'ja': ProcessingConfig(data_directory="data/ja", output_directory="output/ja", ...),
}

for lang, config in configs.items():
    extractor = KnowledgeGraphExtractor(model=triple_generator, config=config)
    extractor.run_extraction()
    extractor.convert_json_to_csv()
    extractor.generate_concept_csv_temp(language=lang)
    extractor.create_concept_csv()

# Step 2: Merge graphs with entity alignment
# (This requires additional entity linking/alignment - see research papers on cross-lingual entity alignment)
```

## Related Documentation

- [Main Example README](readme.md) - Overview of example directory
- [Custom Extraction](example_scripts/custom_extraction/readme.md) - Custom prompts and schemas
- [AutoSchemaKG Main README](../README.md) - Project overview

## Sample Datasets

The `example_data/multilingual_data/` directory contains sample datasets:
- **RomanceOfTheThreeKingdom-zh-CN.json**: Simplified Chinese (三国演义)
- **RomanceOfTheThreeKingdom-zh-HK.json**: Traditional Chinese (三國演義)

Use these as templates for your own multi-language corpora.
