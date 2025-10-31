import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from scripts.config import TRANSLATED_CSV, CLEAN_DESCRIPTIONS_CSV, TRANSLATION_MODEL, DEVICE

try:
    from deep_translator import GoogleTranslator
    HAS_DEEP_TRANSLATOR = True
except ImportError:
    HAS_DEEP_TRANSLATOR = False


def load_m2m100(model_name: str = TRANSLATION_MODEL, device: str = DEVICE):
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def translate_batch(texts, src_lang: str, tgt_lang: str, tokenizer, model, device: str) -> list:
    try:
        tokenizer.src_lang = src_lang
        # Check if target language is supported
        try:
            tgt_lang_id = tokenizer.get_lang_id(tgt_lang)
        except KeyError:
            # Use deep-translator for unsupported languages
            if HAS_DEEP_TRANSLATOR:
                print(f"Language '{tgt_lang}' not in M2M100. Using GoogleTranslator...")
                translator = GoogleTranslator(source=src_lang, target=tgt_lang)
                results = []
                for text in texts:
                    try:
                        translated = translator.translate(text)
                        results.append(translated.strip())
                    except Exception as e:
                        print(f"Translation error for text: {text[:50]}... Error: {e}")
                        results.append("")
                return results
            else:
                print(f"Warning: Language '{tgt_lang}' not supported and deep-translator not available.")
                return [""] * len(texts)
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            generated_tokens = model.generate(**encoded, forced_bos_token_id=tgt_lang_id)
        outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return [o.strip() for o in outputs]
    except Exception as e:
        print(f"Error translating to {tgt_lang}: {e}. Using empty strings as fallback.")
        return [""] * len(texts)


def main():
    df = pd.read_csv(CLEAN_DESCRIPTIONS_CSV)
    tokenizer, model = load_m2m100()

    texts = df["text_en"].astype(str).tolist()
    # Translate to multiple languages
    print("Translating to Spanish...")
    es = translate_batch(texts, src_lang="en", tgt_lang="es", tokenizer=tokenizer, model=model, device=DEVICE)
    print("Translating to French...")
    fr = translate_batch(texts, src_lang="en", tgt_lang="fr", tokenizer=tokenizer, model=model, device=DEVICE)
    print("Translating to Hindi...")
    hi = translate_batch(texts, src_lang="en", tgt_lang="hi", tokenizer=tokenizer, model=model, device=DEVICE)
    print("Translating to Telugu...")
    te = translate_batch(texts, src_lang="en", tgt_lang="te", tokenizer=tokenizer, model=model, device=DEVICE)

    df["text_es"] = es
    df["text_fr"] = fr
    df["text_hi"] = hi
    df["text_te"] = te
    df.to_csv(TRANSLATED_CSV, index=False)
    print(f"Wrote translations: {TRANSLATED_CSV} ({len(df)} rows)")
    print("Languages: EN, ES, FR, HI, TE")


if __name__ == "__main__":
    main()

