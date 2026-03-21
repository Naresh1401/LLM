import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules', 'financial_bot'))

from financial_bot.template import PromptTemplate, get_llm_template, templates

# Test 1: Template registration
assert 'falcon' in templates, "falcon template should be registered"
print("✅ Falcon template registered")

# Test 2: Template properties
tmpl = get_llm_template('falcon')
assert 'question' in tmpl.input_variables
assert 'answer' in tmpl.input_variables
print(f"✅ Template input_variables: {tmpl.input_variables}")

# Test 3: Format training sample
sample = {
    "user_context": "I am interested in tech stocks",
    "news_context": "NVIDIA reported record Q3 earnings",
    "chat_history": "",
    "question": "Should I invest in NVIDIA?",
    "answer": "Based on recent earnings, NVIDIA shows strong fundamentals."
}
result = tmpl.format_train(sample)
assert "NVIDIA" in result["prompt"]
assert "question" in result["payload"]
print("✅ Training template formats correctly")

# Test 4: Format inference sample
infer_sample = {
    "user_context": "I am a conservative investor",
    "news_context": "Markets are volatile due to Fed rate decisions",
    "question": "What should I do with my portfolio?"
}
result = tmpl.format_infer(infer_sample)
assert "portfolio" in result["prompt"]
assert "answer" not in result["prompt"]  # inference has no answer
print("✅ Inference template formats correctly (no answer leak)")

# Test 5: Test the training data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules', 'training_pipeline'))
import json
data_path = os.path.join(os.path.dirname(__file__), 'modules', 'training_pipeline', 'dataset', 'training_data.json')
with open(data_path) as f:
    training_data = json.load(f)
assert len(training_data) > 0
print(f"✅ Training dataset loaded: {len(training_data)} samples")
first = training_data[0]
assert 'about_me' in first or 'instruction' in first or 'question' in first
print(f"✅ Training data sample keys: {list(first.keys())[:5]}")

print()
print("All LLM financial bot checks passed!")
