from helper_functions import *

squad_v1, squad_v2  = get_squad()

# prompt to LLMS
# you will be given a context inside <CONTEXT> tags and a question inside <QUESTION> tags. 
# Your task is to answer the question based on the context.
# If the context is not sufficient to answer the question, your answer should be "unanswerable".
# Place your answer inside <ANSWER> tags.

# Set up the QA pipeline and get the raw model and tokenizer
qa_pipeline, model, tokenizer = setup_qa_model()

# Analyze SQuAD v2 samples
results = analyze_squad(qa_pipeline, model, tokenizer, squad_v2['validation'], num_samples=None) #10)

# Save results to a JSON file
with open('squad_v2_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

