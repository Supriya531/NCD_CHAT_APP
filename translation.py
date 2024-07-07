import torch,os
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig,file_utils
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

class TranslationIndic:
    def __init__(self):
        self.BATCH_SIZE = 4
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantization = None
        self.LANGUAGE_CODE_MAP = {
            "Assamese": "asm_Beng",
            "ben": "ben_Beng",
            "Bodo Parja": "bdv_Deva",
            "Dogri": "dgo_Deva",
            "Gujarati": "guj_Gujr",
            "hin": "hin_Deva",
            "Kannada": "kan_Knda",
            "Maithili": "mai_Deva",
            "Malayalam": "mal_Mlym",
            "Marathi": "mar_Deva",
            "Odia": "ory_Orya",
            "Punjabi, Eastern": "pan_Guru",
            "tam": "tam_Taml",
            "tel_Telu": "tel_Telu",
            "Urdu (Devanagari script)": "urd_Deva",
            "Urdu (Arabic script)": "urd_Arab",
            "Urdu (Latin script)": "urd_Latn"
        }

    def initialize_model_and_tokenizer(self, ckpt_dir, direction):
        if self.quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        tokenizer = IndicTransTokenizer(direction=direction)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        )

        if qconfig is None:
            model = model.to(self.DEVICE)
            if self.DEVICE == "cuda":
                model.half()

        model.eval()

        return tokenizer, model

    def split_text(self, text, max_length=512):
        import re

        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def batch_translate(self, input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
        translations = []
        for i in range(0, len(input_sentences), self.BATCH_SIZE):
            batch = input_sentences[i : i + self.BATCH_SIZE]

            # Preprocess the batch and extract entity mappings
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

            # Tokenize the batch and generate input encodings
            inputs = tokenizer(
                batch,
                src=True,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.DEVICE)

            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=512,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode the generated tokens into text
            generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

            # Postprocess the translations, including entity replacement
            translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

            del inputs
            torch.cuda.empty_cache()

        return translations

    def translate_to_english(self, text, src_language):
        model_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
        tokenizer, model = self.initialize_model_and_tokenizer(model_ckpt_dir, "indic-en")
        ip = IndicProcessor(inference=True)

        try:
            translations = self.batch_translate([text], src_language, "eng_Latn", model, tokenizer, ip)
            translated_text = translations[0] if translations else None
        except IndexError:
            print("Translation Error: List index out of range.")
            translated_text = None

        # Flush the models to free the GPU memory
        del tokenizer, model

        return translated_text

    def translate_to_indic(self, text, tgt_language):
        model_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
        tokenizer, model = self.initialize_model_and_tokenizer(model_ckpt_dir, "en-indic")
        ip = IndicProcessor(inference=True)

        chunks = self.split_text(text)

        translations = []
        try:
            for chunk in chunks:
                chunk_translations = self.batch_translate([chunk], "eng_Latn", tgt_language, model, tokenizer, ip)
                if chunk_translations:
                    translations.extend(chunk_translations)
        except IndexError:
            print("Translation Error: List index out of range.")
            translated_text = None
        finally:
            # Flush the models to free the GPU memory
            del tokenizer, model
            torch.cuda.empty_cache()

        translated_text = ' '.join(translations) if translations else None
        return translated_text

        # Print the absolute path of the cache directory being used
        cache_dir = file_utils.default_cache_path
        absolute_cache_dir = os.path.abspath(cache_dir)
        print("Transformers cache directory:", absolute_cache_dir)