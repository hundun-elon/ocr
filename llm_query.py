from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def __init__(self, model_name_or_path=None, device='cpu'):
      self.device = device
      self.model_name_or_path = model_name_or_path
      self._init_model()


def _init_model(self):
# Try to load a Mistral-style model if provided; else fall back to a tiny demo model
      try:
            if self.model_name_or_path:
                  print(f'Loading model from {self.model_name_or_path} (this may take a while)')
                  self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
                  self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map='auto' if torch.cuda.is_available() else None)
                  self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
                  return
      except Exception as e:
            print('Could not load provided model:', e)


# Fallback tiny model for demo (fast on CPU)
fallback = 'sshleifer/tiny-gpt2'
print('Falling back to tiny demo model:', fallback)
self.tokenizer = AutoTokenizer.from_pretrained(fallback)
self.model = AutoModelForCausalLM.from_pretrained(fallback)
self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=-1)


def ask(self, prompt, max_new_tokens=256, **gen_opts):
      gen = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=False, **gen_opts)
      return gen[0]['generated_text']




# Helper: structured extraction prompt
EXTRACTION_PROMPT = '''
You are a parser that extracts these fields from a document text: company_name, registration_number, reporting_date, compliance_statements (short list), evidence_snippets (list of quoted short passages with page numbers). If a field is missing, answer null. Output JSON ONLY.


Document text:
"""
{document_text}
"""


Respond with JSON.
'''




def extract_structured(llm_client, document_text):
      prompt = EXTRACTION_PROMPT.format(document_text=document_text[:25000]) # truncate safely
      out = llm_client.ask(prompt, max_new_tokens=512)
      # The model might emit extra text — attempt to find JSON substring
      import re, json
      m = re.search(r'(\{[\s\S]*\})', out)
      if m:
            try:
                  return json.loads(m.group(1))
            except Exception as e:
                  return {'_parse_error': str(e), 'raw': out}
      return {'_raw_output': out}